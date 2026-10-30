const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

function context() {
    const ctx = vm.createContext({console, document: {addEventListener() {}}, window: {}});
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/localai-models.js'), 'utf8'), ctx);
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/library.js'), 'utf8'), ctx);
    return ctx;
}

test('chunk editor exposes escaped passage direction separately from spoken text', () => {
    const ctx = context();
    const html = ctx.renderLibraryChunkRow('job', {id: 'one', text: 'Spoken words.',
        delivery_instruction: 'Softly. </textarea><script>bad()</script>'}, 'breeze_tts_2', 0);
    assert(html.includes('Voice Direction:'));
    assert(html.includes('class="library-chunk-direction"'));
    assert(html.includes('&lt;/textarea&gt;&lt;script&gt;'));
    assert(!html.includes('<script>bad()'));
    assert(html.includes('rows="3">Spoken words.</textarea>'));
    assert.equal(ctx.getChunkDirection({emotion: 'Legacy cue'}), 'Legacy cue');
    assert.equal(ctx.getChunkDirection({delivery_instruction: '', emotion: 'Legacy cue'}), '');
});

for (const direction of ['Calm, measured delivery.', '']) {
    test(`regenerate sends the edited direction, including explicit clear: ${JSON.stringify(direction)}`, async () => {
        const ctx = context();
        const fields = {
            '.library-chunk-textarea': {value: 'Unchanged words.'},
            '.library-chunk-direction': {value: direction},
            '.library-chunk-engine-select': {value: 'breeze_tts_2'},
        };
        const card = {querySelector: selector => fields[selector] || null};
        const requests = [];
        Object.assign(ctx, {
            setRegenButtonBusy() {}, requestLibraryReviewRestore: async () => {},
            setLibraryRecompileButtonState() {}, updateLibraryChunkStatus() {}, startLibraryChunkRegenWatcher() {},
            alert(message) { throw new Error(message); },
            fetch: async (url, options) => {
                requests.push(JSON.parse(options.body));
                return {ok: true, json: async () => ({success: true})};
            },
        });
        vm.runInContext('libraryVoiceMap = new Map(); chunkReviewModalData = {engine: "breeze_tts_2", chunks: [{id: "one", delivery_instruction: "Old cue"}]};', ctx);
        await ctx.triggerLibraryChunkRegen('job', 'one', {closest: () => card});
        assert.equal(requests.length, 1);
        assert.equal(requests[0].delivery_instruction, direction);
        assert.equal(requests[0].text, 'Unchanged words.');
    });
}
