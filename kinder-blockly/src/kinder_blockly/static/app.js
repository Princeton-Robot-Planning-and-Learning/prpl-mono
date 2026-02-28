/* Blockly workspace setup and program execution. */

var workspace = Blockly.inject('blockly-div', {
    toolbox: document.getElementById('toolbox'),
    trashcan: true,
    scrollbars: true,
});

function loadInitialFrame() {
    document.getElementById('frame-info').textContent = 'Loading environment...';
    fetch('/reset', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({seed: 0}),
    })
    .then(function(resp) { return resp.json(); })
    .then(function(data) {
        if (data.frame) {
            document.getElementById('frame-display').src =
                'data:image/jpeg;base64,' + data.frame;
            document.getElementById('frame-info').textContent =
                'Drag blocks and click Run';
        }
    })
    .catch(function(err) {
        document.getElementById('frame-info').textContent = 'Failed to load: ' + err;
    });
}

loadInitialFrame();

function extractProgram() {
    var blocks = [];
    var topBlocks = workspace.getTopBlocks(true);
    for (var i = 0; i < topBlocks.length; i++) {
        var block = topBlocks[i];
        while (block) {
            var entry = {type: block.type};
            if (block.type === 'move_base_to_target') {
                entry.x = block.getFieldValue('X');
                entry.y = block.getFieldValue('Y');
            }
            blocks.push(entry);
            block = block.getNextBlock();
        }
    }
    return {blocks: blocks};
}

function runProgram() {
    var program = extractProgram();
    if (program.blocks.length === 0) {
        document.getElementById('status').textContent = 'No blocks to run.';
        return;
    }

    var btn = document.getElementById('run-btn');
    btn.disabled = true;
    document.getElementById('status').textContent = 'Running...';
    document.getElementById('frame-info').textContent = '';

    fetch('/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({program: program, seed: 0}),
    })
    .then(function(resp) { return resp.json(); })
    .then(function(data) {
        btn.disabled = false;
        if (data.error) {
            document.getElementById('status').textContent = 'Error: ' + data.error;
            return;
        }
        document.getElementById('status').textContent = 'Done!';
        playFrames(data.frames || []);
    })
    .catch(function(err) {
        btn.disabled = false;
        document.getElementById('status').textContent = 'Error: ' + err;
    });
}

function playFrames(frames) {
    if (frames.length === 0) {
        document.getElementById('frame-info').textContent = 'No frames.';
        return;
    }
    var img = document.getElementById('frame-display');
    var idx = 0;
    var info = document.getElementById('frame-info');

    function showFrame() {
        img.src = 'data:image/jpeg;base64,' + frames[idx];
        info.textContent = 'Frame ' + (idx + 1) + ' / ' + frames.length;
        idx++;
        if (idx < frames.length) {
            setTimeout(showFrame, 100);
        }
    }
    showFrame();
}
