"""Utility functions."""

import ast
import builtins
import hashlib
from typing import Any, Callable


def consistent_hash(obj: Any) -> int:
    """A hash function that is consistent between sessions, unlike hash()."""
    obj_str = repr(obj)
    obj_bytes = obj_str.encode("utf-8")
    hash_hex = hashlib.sha256(obj_bytes).hexdigest()
    hash_int = int(hash_hex, 16)
    # Mimic Python's built-in hash() behavior by returning a 64-bit signed int.
    # This makes it comparable to hash()'s output range.
    return hash_int if hash_int < 2**63 else hash_int - 2**6


class Scope:
    """Represents a variable scope in Python code, tracking defined, global,
    and nonlocal variables."""

    def __init__(self, parent: "Scope | None" = None) -> None:
        """Initialize a new Scope instance."""
        self.parent = parent
        self.defined: set[str] = set()
        self.globals: set[str] = set()
        self.nonlocals: set[str] = set()

    def is_defined(self, name: str) -> bool:
        """Check if a variable is defined in the current or parent scopes."""
        if name in self.defined:
            return True
        if name in self.globals:
            root = self
            while root.parent:
                root = root.parent
            return name in root.defined
        if name in self.nonlocals and self.parent:
            return self.parent.is_defined(name)
        return self.parent.is_defined(name) if self.parent else False


class UndefinedVisitor(ast.NodeVisitor):
    """AST visitor to identify undefined variable names in Python code."""

    DEFAULT_ALLOWED_IMPORTS = {
        "math",
        "statistics",
        "itertools",
        "functools",
        "collections",
        "re",
        "json",
        "typing",
        "dataclasses",
        "heapq",
        "bisect",
        "operator",
        "random",
    }

    def __init__(
        self,
        provided_globals: set[str] | None = None,
        allowed_imports: set[str] | None = None,
    ) -> None:
        """Initialize the UndefinedVisitor."""
        self.issues: set[str] = set()
        self.scope: Scope = Scope()
        self.provided: set[str] = set(provided_globals or [])
        self.allowed_imports: set[str] = allowed_imports or self.DEFAULT_ALLOWED_IMPORTS
        # builtins available by default
        self.scope.defined |= set(dir(builtins))
        # map alias -> top-level module (e.g, self.alias_to_mod = {"np": "numpy"})
        self.alias_to_mod: dict[str, str] = {}

    def push(self) -> None:
        """Push a new scope."""
        self.scope = Scope(self.scope)

    def pop(self) -> None:
        """Pop the current scope."""
        self.scope = self.scope.parent  # type: ignore

    # --- utilities ---
    def _define_target(self, target: ast.AST) -> None:
        """Define a variable or variables in the current scope."""
        # x = 1 -> defines "x"
        if isinstance(target, ast.Name):
            self.scope.defined.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._define_target(elt)

    def _load_name(self, name_node: ast.Name) -> None:
        """Check if a variable is defined; if not, add it to issues."""
        name = name_node.id
        if not (self.scope.is_defined(name) or name in self.provided):
            self.issues.add(name)

    # ---------- visitors ----------
    def visit_Name(self, node: ast.Name) -> None:
        """Visit a Name node."""
        # y + 1 -> visit_Name("y", Load) -> flagged if not defined
        if isinstance(node.ctx, ast.Load):
            self._load_name(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Handle assignment statements in the AST."""
        # x = 5 -> defines "x"
        self.visit(node.value)
        for t in node.targets:
            self._define_target(t)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handle annotated assignment statements in the AST."""
        # x: int = 10 -> defines "x"
        if node.value:
            self.visit(node.value)
        self._define_target(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Handle augmented assignment statements in the AST."""
        # x += 1 -> requires "x" defined first
        self.visit(node.target)
        self.visit(node.value)
        self._define_target(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        """Handle named expressions (walrus operator) in the AST."""
        # (x := 5) -> defines "x
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self.scope.defined.add(node.target.id)

    def visit_For(self, node: ast.For) -> None:
        """Handle for-loops in the AST."""
        # for i in range(3): ... -> defines "i"
        self.visit(node.iter)
        self._define_target(node.target)
        self.push()
        for s in node.body:
            self.visit(s)
        self.pop()
        for s in node.orelse:
            self.visit(s)

    def visit_With(self, node: ast.With) -> None:
        """Handle with-statements in the AST."""
        # with open("x.txt") as f: ... -> defines "f"
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                self._define_target(item.optional_vars)
        self.push()
        for s in node.body:
            self.visit(s)
        self.pop()

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        """Handle exception handlers in the AST."""
        # except Exception as e: ... -> defines "e"
        if node.name:
            # py>=3.11 can be ast.ExceptHandler.name as str or ast.Name older
            name = node.name if isinstance(node.name, str) else node.name.id
            self.scope.defined.add(name)
        self.push()
        for s in node.body:
            self.visit(s)
        self.pop()

    def visit_Match(self, node: ast.Match):
        """Handle match-statements (Python 3.10+) in the AST."""
        #   match x: ... case y: ... -> defines "y"
        self.visit(node.subject)
        for case in node.cases:
            self.push()
            self.visit(case.pattern)
            if case.guard:
                self.visit(case.guard)
            for s in case.body:
                self.visit(s)
            self.pop()

    def visit_comprehension(self, comp: ast.comprehension):
        """Handle comprehensions (e.g., list comprehensions) in the AST."""
        # [x for x in data if cond(x)] -> defines "x" in inner scope
        self.visit(comp.iter)
        self._define_target(comp.target)
        for if_ in comp.ifs:
            self.visit(if_)

    def visit_ListComp(self, node: ast.ListComp):
        """Handle list comprehensions in the AST."""
        # squares = [x*x for x in range(5)]
        self.push()
        for gen in node.generators:
            self.visit_comprehension(gen)
        self.visit(node.elt)
        self.pop()

    def visit_DictComp(self, node: ast.DictComp):
        """Handle dictionary comprehensions in the AST."""
        # {x: x**2 for x in range(3)}
        self.push()
        for gen in node.generators:
            self.visit_comprehension(gen)
        self.visit(node.key)
        self.visit(node.value)
        self.pop()

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        """Handle generator expressions in the AST."""
        # (x for x in range(3))
        self.push()
        for gen in node.generators:
            self.visit_comprehension(gen)
        self.visit(node.elt)
        self.pop()

    def visit_Global(self, node: ast.Global):
        """Handle global variable declarations in the AST."""
        # global x -> declares x as global
        self.scope.globals |= set(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal):
        """Handle nonlocal variable declarations in the AST."""
        # nonlocal x -> refers to x in outer, non-global, scope
        self.scope.nonlocals |= set(node.names)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Handle function definitions in the AST."""
        # def f(a): b = a -> defines "f" and "a", "b" inside inner scope

        self.scope.defined.add(node.name)  # function name binds in outer scope
        # new inner scope for params + body
        self.push()
        args = node.args
        for a in args.args + args.kwonlyargs:
            self.scope.defined.add(a.arg)
        if args.vararg:
            self.scope.defined.add(args.vararg.arg)
        if args.kwarg:
            self.scope.defined.add(args.kwarg.arg)
        for s in node.body:
            self.visit(s)
        self.pop()

    # to handle both normal and async versions with the same logic
    visit_AsyncFor: Callable[[ast.AsyncFor], None] = visit_For  # type: ignore
    visit_AsyncWith: Callable[[ast.AsyncWith], None] = visit_With  # type: ignore
    visit_SetComp: Callable[[ast.SetComp], None] = visit_ListComp  # type: ignore
    visit_AsyncFunctionDef: Callable[
        [ast.AsyncFunctionDef], None
    ] = visit_FunctionDef  # type: ignore

    def visit_Lambda(self, node: ast.Lambda):
        """Handle lambda expressions in the AST."""
        # lambda x: x + 1 -> defines "x"
        self.push()
        args = node.args
        for a in args.args + args.kwonlyargs:
            self.scope.defined.add(a.arg)
        if args.vararg:
            self.scope.defined.add(args.vararg.arg)
        if args.kwarg:
            self.scope.defined.add(args.kwarg.arg)
        self.visit(node.body)
        self.pop()

    def visit_ClassDef(self, node: ast.ClassDef):
        """Handle class definitions in the AST."""
        # class A: pass -> defines "A"
        # class name binds in enclosing scope
        self.scope.defined.add(node.name)
        self.push()
        for s in node.body:
            self.visit(s)
        self.pop()

    def visit_Import(self, node: ast.Import):
        """Handle import statements in the AST."""

        for alias in node.names:
            mod = alias.name.split(".")[0]
            asname = alias.asname or mod
            self.scope.defined.add(asname)
            self.alias_to_mod[asname] = mod

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Handle from-import statements in the AST."""
        for alias in node.names:
            asname = alias.asname or alias.name
            self.scope.defined.add(asname)


def find_undefined_names(
    source: str, *, provided_globals: set[str] | None = None
) -> set[str]:
    """Identify undefined variable names in the given Python source code."""
    tree = ast.parse(source)
    v = UndefinedVisitor(provided_globals=provided_globals or set())
    v.visit(tree)
    return v.issues  # set of strings
