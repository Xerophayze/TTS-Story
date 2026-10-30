const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const read = f => fs.readFileSync(path.join(__dirname, '..', f), 'utf8');
const source = read('static/js/library.js');

test('candidate assignment uses the active speaker dropdown and survives a catalog rebuild', async () => {
    const main = read('static/js/main.js');
    const makeSelect = speaker => ({dataset: {}, value: '', options: [], closest: () => ({dataset: {speaker}}),
        add(option) {this.options.push(option);}, dispatchEvent() {}});
    const selected = makeSelect('maya-female');
    const other = makeSelect('narrator');
    const profiles = {};
    const context = vm.createContext({window: {isGeminiTtsEngine: () => true, geminiTtsVoices: []},
        getSelectedJobEngine: () => 'gemini_tts', document: {querySelectorAll: () => [selected, other]},
        updateSpeakerProfileEntry: (speaker, values) => {profiles[speaker] = {...profiles[speaker], ...values};},
        Option: function(label, value) {this.textContent = label; this.value = value;}, Event: function() {}});
    vm.runInContext(main.slice(main.indexOf('async function assignGeminiSpeakerVoice('), main.indexOf('function renderSpeakerProfileSummary(')), context);
    await context.assignGeminiSpeakerVoice('maya-female', 'voice_maya', 'Maya');
    assert.equal(selected.value, 'voice_maya');
    assert.equal(other.value, '');
    assert.equal(profiles['maya-female'].selected_voice_name, 'Maya');
    assert.equal(profiles['maya-female'].selected_voice_path, '');
    selected.value = ''; selected.options = [];
    await context.assignGeminiSpeakerVoice('maya-female', 'voice_maya', 'Maya');
    assert.equal(selected.value, 'voice_maya');
    await assert.rejects(context.assignGeminiSpeakerVoice('missing', 'voice_a', 'A'), /no longer available/);
});

test('using a Gemini candidate offers a safe engine switch without racing Analyze Text', async () => {
    const main = read('static/js/main.js');
    const engine = {value: 'breeze_api', dispatchEvent() { throw Error('Must not trigger asynchronous reanalysis'); }};
    const select = {dataset: {speaker: 'actor'}, value: '', options: [], add(v) {this.options.push(v);}, dispatchEvent() {}};
    let consent = false, captured = 0, updates = 0, catalogs = 0;
    const profiles = {};
    const context = vm.createContext({window: {
        isGeminiTtsEngine: v => ['gemini_tts', 'gemini_tts_lite'].includes(v), geminiTtsVoices: [],
        confirm: () => consent, loadGeminiTtsCatalog: async () => { catalogs++; }},
        getSelectedJobEngine: () => engine.value, captureCompatibleVoiceSamples: () => {captured++;},
        updateEngineUI: value => { assert.equal(value, 'gemini_tts_lite'); updates++; },
        updateModeIndicator() {}, populateDefaultVoiceSelect() {},
        updateSpeakerProfileEntry: (speaker, values) => {profiles[speaker] = {...profiles[speaker], ...values};},
        document: {getElementById: () => engine, querySelectorAll: () => [select]},
        Option: function(label, value) {this.value = value;}, Event: function() {}});
    vm.runInContext(main.slice(main.indexOf('async function assignGeminiSpeakerVoice('), main.indexOf('function renderSpeakerProfileSummary(')), context);
    assert.equal(await context.assignGeminiSpeakerVoice('actor', 'voice_a', 'Actor', 'gemini_tts_lite'), false);
    assert.equal(engine.value, 'breeze_api'); assert.equal(select.value, ''); assert.equal(catalogs, 0);
    consent = true;
    assert.equal(await context.assignGeminiSpeakerVoice('actor', 'voice_a', 'Actor', 'gemini_tts_lite'), true);
    assert.equal(engine.value, 'gemini_tts_lite'); assert.equal(select.value, 'voice_a');
    assert.equal(captured, 1); assert.equal(updates, 1); assert.equal(catalogs, 1);
    assert.equal(profiles.actor.selected_voice_id, 'voice_a');
});

test('speaker Gemini design is embedded and retains audition after selection', () => {
    const main = read('static/js/main.js');
    const designer = read('static/js/gemini-tts.js');
    assert.match(main, /openGeminiVoiceDesigner\(\{\s*container: summary,\s*candidateCount: candidateCountInput/);
    assert.match(designer, /document\.createElement\(container \? 'section' : 'dialog'\)/);
    assert.match(designer, /if \(container\) \{\s*container\.append\(dialog\)/);
    assert.match(designer, /else if \(container\) \{\s*status\.textContent = `Selected voice:/);
    assert.match(main, /geminiDesigner\?\.dataset\.busy === 'true'/);
    assert.match(main, /geminiDesigner\?\.geminiDispose\?\.\(\)/);
    assert.match(designer, /if \(existing\) \{ existing\.scrollIntoView/);
});

test('speaker design selector supports Gemini independently of narration engine and local installs', () => {
    const main = read('static/js/main.js');
    assert.match(main, /<option value="gemini" data-voice-design-option/);
    assert.match(main, /if \(voiceDesignEngineSelect\?\.value === 'gemini'\) openGeminiDesigner\(false\)\?\.geminiGenerate\(\)/);
    const context = vm.createContext({window: {}});
    vm.runInContext(main.slice(main.indexOf('function normalizeVoiceDesignEngine('), main.indexOf('function getVoiceDesignEngineConfig(')), context);
    assert.equal(context.normalizeVoiceDesignEngine('gemini'), 'gemini');
    assert.equal(context.voiceDesignEngineAvailable('gemini'), true);
    assert.equal(context.voiceDesignEngineAvailabilityKnown('gemini'), true);
    assert.equal(context.voiceDesignEngineAvailable('qwen3'), false);
    assert.equal(context.voiceDesignEngineAvailable('breeze'), false);
});

for (const engine of ['gemini_tts', 'gemini_tts_lite']) {
    test(`${engine} bulk review loads Google presets and hosted designs, not local samples`, async () => {
        const voices = new Map();
        const select = {value: '', options: [], appendChild(v) {this.options.push(v);}};
        const context = vm.createContext({
            console, window: {localAIEngineChoice: engine => ({engine})},
            document: {createElement: () => ({dataset: {}, style: {}})},
            updateLocalAIManualFields() {}, getMinDuration: () => 0,
            libraryVoiceFilters: {}, libraryVoiceMap: voices,
            getLibraryLanguageDisplayName: v => v,
            getLibraryCloudCatalog: async name => {
                assert.equal(name, engine);
                return {success: true, voices: [{short_name: 'Kore', display_name: 'Kore'}, {short_name: 'voice_actor123', display_name: 'Designed Actor'}]};
            },
            getVoicePromptsCached: () => {throw Error('Must not load local samples');},
            getVoicesCached: () => {throw Error('Must not load Kokoro voices');},
        });
        vm.runInContext(source.slice(source.indexOf('function mapLibraryCloudVoices('), source.indexOf('function appendLocalAIFreeformOption(')), context);
        vm.runInContext(source.slice(source.indexOf('    async function populateBulkVoiceSelect('), source.indexOf('    const chunkById')), context);
        await context.populateBulkVoiceSelect(select, 'narrator', engine);
        assert.ok(select.options.some(v => v.value === 'Kore'));
        assert.equal(voices.get('Kore').isPrompt, false);
        assert.ok(select.options.some(v => v.value === 'voice_actor123'));
        assert.equal(voices.get('voice_actor123').isPrompt, false);
    });
}

test('both model choices are present in main and all library regeneration menus', () => {
    assert.equal((read('templates/index.html').match(/value="gemini_tts"/g) || []).length, 4);
    assert.equal((source.match(/value="gemini_tts_lite"/g) || []).length, 3);
    assert.match(read('static/js/settings.js'), /collectGeminiTtsSettings/);
    assert.match(read('templates/index.html'), /id="gemini-tts-api-key"/);
});
