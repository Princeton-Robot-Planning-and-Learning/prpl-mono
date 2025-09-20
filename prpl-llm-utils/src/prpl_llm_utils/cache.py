"""Methods for saving and loading model responses."""

import abc
import json
import logging
import sqlite3
from pathlib import Path

import imagehash

from prpl_llm_utils.structs import Query, Response


class ResponseNotFound(Exception):
    """Raised during cache lookup if a response is not found."""


class PretrainedLargeModelCache(abc.ABC):
    """Base class for model caches."""

    @abc.abstractmethod
    def try_load_response(self, query: Query, model_id: str) -> Response:
        """Load a response or raise ResponseNotFound."""

    @abc.abstractmethod
    def save(self, query: Query, model_id: str, response: Response) -> None:
        """Save the response for the query."""


class FilePretrainedLargeModelCache(PretrainedLargeModelCache):
    """A cache that saves and loads from individual files."""

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(exist_ok=True)

    def _get_cache_dir_for_query(self, query: Query, model_id: str) -> Path:
        query_id = query.get_readable_id()
        cache_foldername = f"{model_id}_{query_id}"
        cache_folderpath = self._cache_dir / cache_foldername
        cache_folderpath.mkdir(exist_ok=True)
        return cache_folderpath

    def try_load_response(self, query: Query, model_id: str) -> Response:
        cache_dir = self._get_cache_dir_for_query(query, model_id)
        if not (cache_dir / "prompt.txt").exists():
            raise ResponseNotFound
        # Load the saved completions.
        completion_file = cache_dir / "completion.txt"
        with open(completion_file, "r", encoding="utf-8") as f:
            completion = f.read()
        # Load the metadata.
        metadata_file = cache_dir / "metadata.json"
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        # Create the response.
        response = Response(completion, metadata)
        logging.debug(f"Loaded model response from {cache_dir}.")
        return response

    def save(self, query: Query, model_id: str, response: Response) -> None:
        cache_dir = self._get_cache_dir_for_query(query, model_id)
        # Cache the text prompt.
        prompt_file = cache_dir / "prompt.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(query.prompt)
        # Cache the image prompt if it exists.
        if query.imgs is not None:
            imgs_folderpath = cache_dir / "imgs"
            imgs_folderpath.mkdir(exist_ok=True)
            for i, img in enumerate(query.imgs):
                filename_suffix = str(i) + ".jpg"
                img.save(imgs_folderpath / filename_suffix)
        # Cache the text response.
        completion_file = cache_dir / "completion.txt"
        with open(completion_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        # Cache the metadata.
        metadata_file = cache_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(response.metadata, f)
        logging.debug(f"Saved model response to {cache_dir}.")


class SQLite3PretrainedLargeModelCache(PretrainedLargeModelCache):
    """A cache that uses a SQLite3 database."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database_path.parent.mkdir(exist_ok=True)
        self._initialized = False
        self._hyperparameter_keys: set[str] | None = None

    def _get_query_hash(self, query: Query, model_id: str) -> str:
        """Get a unique hash for the query and model combination."""
        query_id = query.get_id()
        return f"{model_id}_{hash(query_id)}"

    def _ensure_initialized(self, query: Query) -> None:
        """Initialize the database with the required tables and columns."""
        if self._initialized:
            # Verify hyperparameter keys are consistent.
            if query.hyperparameters is not None:
                current_keys = set(query.hyperparameters.keys())
                if self._hyperparameter_keys is not None:
                    assert current_keys == self._hyperparameter_keys, (
                        f"Hyperparameter changed from {self._hyperparameter_keys} "
                        f"to {current_keys}. All queries must use the same."
                    )
                else:
                    self._hyperparameter_keys = current_keys
            return

        with sqlite3.connect(self._database_path) as conn:
            # Create base table.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    query_hash TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    images_hash TEXT,
                    completion TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
            """
            )

            # Add hyperparameter columns if present.
            if query.hyperparameters is not None:
                self._hyperparameter_keys = set(query.hyperparameters.keys())
                for key in self._hyperparameter_keys:
                    try:
                        conn.execute(f"ALTER TABLE responses ADD COLUMN {key} TEXT")
                    except sqlite3.OperationalError:
                        # Column already exists, ignore.
                        pass

            conn.commit()
            self._initialized = True

    def try_load_response(self, query: Query, model_id: str) -> Response:
        self._ensure_initialized(query)
        query_hash = self._get_query_hash(query, model_id)

        with sqlite3.connect(self._database_path) as conn:
            cursor = conn.execute(
                "SELECT completion, metadata FROM responses WHERE query_hash = ?",
                (query_hash,),
            )
            result = cursor.fetchone()

            if result is None:
                raise ResponseNotFound

            completion, metadata_json = result
            metadata = json.loads(metadata_json)
            response = Response(completion, metadata)
            logging.debug(
                f"Loaded model response from SQLite for query hash {query_hash}."
            )
            return response

    def save(self, query: Query, model_id: str, response: Response) -> None:
        self._ensure_initialized(query)
        query_hash = self._get_query_hash(query, model_id)

        # Prepare the data for storage.
        images_hash = None
        if query.imgs is not None:
            img_hash_list = [str(imagehash.phash(img)) for img in query.imgs]
            images_hash = json.dumps(img_hash_list)

        metadata_json = json.dumps(response.metadata)

        # Build base columns and values.
        columns = [
            "query_hash",
            "model_id",
            "prompt",
            "images_hash",
            "completion",
            "metadata",
        ]
        values = [
            query_hash,
            model_id,
            query.prompt,
            images_hash,
            response.text,
            metadata_json,
        ]

        # Add hyperparameters if present.
        if query.hyperparameters is not None:
            columns.extend(query.hyperparameters.keys())
            values.extend(json.dumps(value) for value in query.hyperparameters.values())

        placeholders = ["?"] * len(columns)
        sql = f"""
            INSERT OR REPLACE INTO responses 
            ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
        """

        with sqlite3.connect(self._database_path) as conn:
            conn.execute(sql, values)
            conn.commit()

        logging.debug(
            f"Saved model response to SQLite database for query hash {query_hash}."
        )
