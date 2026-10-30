const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require('node:path').join(__dirname, '../static/js/library.js'), 'utf8');

test('speaker regeneration menu includes both Breeze engines', () => {
    const menu = source.slice(source.indexOf('// Populate bulk speaker engine selects'), source.indexOf('// Also populate bulk speaker voice selects'));
    assert.match(menu, /value="breeze_tts_2"/);
    assert.match(menu, /value="breeze_api"/);
    assert.match(menu, /normalizedSelectedEngine.includes\('breeze'\)/);
});

test('chunk regeneration and alternate-word preview menus also include both Breeze engines', () => {
    const chunkMenu = source.slice(source.indexOf('// Populate chunk engine selects'), source.indexOf('// Populate bulk speaker engine selects'));
    const alternateWordMenu = source.slice(source.indexOf('id="lib-awr-preview-engine"'), source.indexOf('</select>', source.indexOf('id="lib-awr-preview-engine"')));
    for (const menu of [chunkMenu, alternateWordMenu]) {
        assert.match(menu, /value="breeze_tts_2"/);
        assert.match(menu, /value="breeze_api"/);
    }
});

test('completed regeneration refreshes the text summary without creating a media player', () => {
    const preview = {}, button = {setAttribute() {}};
    const cached = {id: '0-0', text: 'Old text'};
    const card = {querySelector: selector => selector === '.library-chunk-preview' ? preview : selector === '.library-chunk-play' ? button : null};
    const context = vm.createContext({document: {querySelector: () => card}, chunkReviewModalData: {chunks: [cached]}});
    const start = source.indexOf('function updateLibraryCompletedChunk(');
    vm.runInContext(source.slice(start, source.indexOf('\nfunction ', start + 10)), context);
    context.updateLibraryCompletedChunk('0-0', {file_url: '/test.wav', text: 'New <literal> text'});
    assert.equal(preview.textContent, 'New <literal> text');
    assert.equal(cached.text, 'New <literal> text');
});

for (const engine of ['breeze_tts_2', 'breeze_api']) {
    test(`${engine} lists shared samples and sends sample paths for bulk regeneration`, async () => {
        const voiceMap = new Map();
        const requests = [];
        const select = { value: 'sample.wav', options: [], appendChild(v) { this.options.push(v); } };
        const card = { querySelector: s => s === '.bulk-speaker-voice-select' ? select : null };
        const context = vm.createContext({
            console, setTimeout: () => 0,
            window: { localAIEngineChoice: engine => ({ engine }), applyLocalAIModelToRegenRequest() {} },
            document: { createElement: () => ({ dataset: {}, style: {} }), getElementById: () => ({}), querySelector: () => null },
            updateLocalAIManualFields() {}, getMinDuration: () => 0,
            libraryVoiceFilters: {}, libraryVoiceMap: voiceMap, libraryChunkVoiceOverrides: {},
            getLibraryLanguageDisplayName: v => v,
            getVoicePromptsCached: async () => ({ success: true, prompts: [{ path: 'sample.wav', display: 'My sample', transcript: 'Preview words.', duration_seconds: 12 }] }),
            fetch: async (url, options) => {
                if (url === '/api/breeze-api/catalog') return { ok: true, json: async () => ({ success: true, voices: [{ voice_id: 'remote-id', name: 'Remote' }] }) };
                requests.push(JSON.parse(options.body));
                return { ok: true, json: async () => ({ success: true }) };
            },
            setLibraryRecompileButtonState() {}, updateLibraryChunkStatus() {},
        });
        const start = source.indexOf('    async function populateBulkVoiceSelect(');
        const end = source.indexOf('    const chunkById', start);
        vm.runInContext(source.slice(start, end), context);
        await context.populateBulkVoiceSelect(select, 'narrator', engine);
        assert.ok(select.options.some(v => v.value === 'sample.wav'));
        assert.equal(voiceMap.get('sample.wav').transcript, 'Preview words.');
        const regenStart = source.indexOf('async function triggerBulkSpeakerRegen(');
        const regenEnd = source.indexOf('\nasync function ', regenStart + 1);
        vm.runInContext(source.slice(regenStart, regenEnd), context);
        const button = { closest: () => card };
        await context.triggerBulkSpeakerRegen('job', 'narrator', [{ id: 'one', speaker: 'narrator', text: 'Hello.' }], engine, button, { skipRestore: true });
        assert.equal(requests[0].engine, engine);
        assert.equal(requests[0].voice.audio_prompt_path, 'sample.wav');
        assert.equal(requests[0].voice.extra.prompt_text, 'Preview words.');
        if (engine === 'breeze_api') {
            assert.ok(select.options.some(v => v.value === 'remote-id'));
            select.value = 'remote-id';
            await context.triggerBulkSpeakerRegen('job', 'narrator', [{ id: 'two', speaker: 'narrator', text: 'Hello.' }], engine, button, { skipRestore: true });
            assert.equal(requests[1].voice.voice, 'remote-id');
            assert.equal(requests[1].voice.audio_prompt_path, undefined);
            assert.equal(requests[1].voice.lang_code, '');
        }
    });
}
