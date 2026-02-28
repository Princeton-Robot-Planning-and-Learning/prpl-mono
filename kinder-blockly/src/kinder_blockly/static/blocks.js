/* Custom Blockly block definitions for KinDER skills. */

Blockly.Blocks['move_base_to_target'] = {
    init: function() {
        this.appendDummyInput()
            .appendField('Move base to')
            .appendField('x')
            .appendField(new Blockly.FieldNumber(0, -2, 2, 0.1), 'X')
            .appendField('y')
            .appendField(new Blockly.FieldNumber(0, -2, 2, 0.1), 'Y');
        this.setPreviousStatement(true, null);
        this.setNextStatement(true, null);
        this.setColour(210);
        this.setTooltip('Move the robot base to the given (x, y) position.');
    }
};
