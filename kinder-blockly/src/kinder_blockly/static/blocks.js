/* Custom Blockly block definitions for KinDER skills. */

Blockly.Blocks['move_base_to_target'] = {
    init: function() {
        this.appendDummyInput()
            .appendField('Move base to target');
        this.setPreviousStatement(true, null);
        this.setNextStatement(true, null);
        this.setColour(210);
        this.setTooltip('Move the robot base to the target position.');
    }
};
