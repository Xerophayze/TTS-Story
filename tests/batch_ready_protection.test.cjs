const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require('node:path').join(__dirname, '../static/js/main.js'), 'utf8');
function load(context, start, end) {
    vm.runInContext(source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start))), context);
}
function setup() {
    const requests = [], notices = [], checkpoints = [], processed = [];
    const profiles = {
        ready: {name: 'ready', description: 'Approved profile', voice: 'Alto', voice_candidate_count: 4,
            voice_design_engine: 'gemini', selected_voice_id: 'voice_approved', selected_voice_path: 'approved.wav',
            gemini_voice_audition: {selected_id: 'voice_approved', candidates: [{id: 'voice_approved'}]}},
        actor: {name: 'actor', description: 'Old profile', voice: 'Baritone', selected_voice_id: 'voice_actor',
            gemini_voice_audition: {candidates: [{id: 'voice_pending'}]}, voice_candidate_count: 2},
    };
    const fields = {'input-text': {value: 'Manuscript'}, 'speaker-batch-candidate-count': {value: '2'},
        'speaker-batch-gemini-model': {value: 'gemini_tts_lite'}};
    const context = vm.createContext({profiles, speakerProfiles: profiles, speakerReadyState: {ready: true},
        currentStats: {speakers: ['ready', 'actor']}, activeProjectId: 'book', latestGeminiBookTitle: '', activeSpeakerModal: null,
        bulkVoiceGenerationState: null, window: {loadGeminiTtsCatalog: async () => {requests.push('catalog');}},
        document: {getElementById: id => fields[id]}, console,
        normalizeSpeakerKey: v => v.replace(/[^a-z0-9]/gi, '').toLowerCase(), normalizeVoiceDesignEngine: v => v || 'qwen3',
        findSpeakerProfile: name => ({profile: profiles[name], matchKey: name}),
        buildLocalVoiceDesignPrompt: (_, voice, prompt) => prompt || voice,
        showNotification: message => notices.push(message),
        fetch: async (_, request) => {requests.push(JSON.parse(request.body)); return {json: async () => ({success: true,
            profiles: {actor: {name: 'actor', description: 'New profile', voice: 'Tenor'}, ready: {description: 'SHOULD NOT APPLY'}}})};},
        persistBulkVoiceGenerationState: async () => checkpoints.push(JSON.parse(JSON.stringify(context.bulkVoiceGenerationState))),
        generateSpeakerVoicePromptBatch: async (speaker, _, __, engine, options) => {processed.push({speaker, engine, options}); return {success: true};},
        buildBatchVoiceName: (_, name) => name, getVoiceDesignEngineConfig: () => ({label: 'Gemini'}),
    });
    load(context, 'function shouldSkipReadySpeaker(', 'async function persistBulkVoiceGenerationState(');
    load(context, 'function updateSpeakerProfileEntry(', 'function findGeneratedSpeakerProfile(');
    load(context, 'async function fetchSpeakerProfiles(', 'function readLegacyBrowserProjects(');
    load(context, 'async function runBatchVoiceGeneration(', 'function buildBatchVoiceName(');
    return {context, profiles, requests, notices, checkpoints, processed, fields};
}
test('bulk profiles target unready speakers and preserve approved casting and pending auditions', async () => {
    const h = setup(), before = JSON.stringify(h.profiles.ready), pending = JSON.stringify(h.profiles.actor.gemini_voice_audition);
    await h.context.fetchSpeakerProfiles();
    assert.deepEqual(h.requests[0].speakers, ['actor']);
    assert.equal(JSON.stringify(h.profiles.ready), before);
    assert.equal(h.profiles.actor.description, 'New profile');
    assert.equal(h.profiles.actor.selected_voice_id, 'voice_actor');
    assert.equal(JSON.stringify(h.profiles.actor.gemini_voice_audition), pending);
});
test('all-ready profile and voice batches make no provider calls', async () => {
    const h = setup(), before = JSON.stringify(h.profiles);
    h.context.speakerReadyState.actor = true;
    await h.context.fetchSpeakerProfiles();
    const summary = {};
    await h.context.runBatchVoiceGeneration('', {}, {}, {completeSummary: summary}, {designEngine: 'gemini'});
    assert.deepEqual(h.requests, []);
    assert.deepEqual(h.processed, []);
    assert.equal(JSON.stringify(h.profiles), before);
    assert.equal(h.context.bulkVoiceGenerationState.skipped_count, 2);
    assert.match(summary.textContent, /2 ready speakers skipped/);
});
test('a speaker marked ready while profiles are in flight is protected', async () => {
    const h = setup(), before = JSON.stringify(h.profiles.actor);
    const fetch = h.context.fetch;
    h.context.fetch = async (...args) => {h.context.speakerReadyState.actor = true; return fetch(...args);};
    await h.context.fetchSpeakerProfiles();
    assert.equal(JSON.stringify(h.profiles.actor), before);
});
test('profile results cannot leak into another project and partial responses retain old profiles', async () => {
    const h = setup(), before = JSON.stringify(h.profiles);
    h.context.fetch = async () => {h.context.activeProjectId = 'other'; return {json: async () => ({success: true, profiles: {actor: {description: 'Wrong project'}}})};};
    await h.context.fetchSpeakerProfiles();
    assert.equal(JSON.stringify(h.profiles), before);
    h.context.fetch = async () => ({json: async () => ({success: true, profiles: {}})});
    await h.context.fetchSpeakerProfiles();
    assert.equal(JSON.stringify(h.profiles), before);
});
test('resume preserves explicit ready override and rechecks ready flags before remaining speakers', async () => {
    const h = setup();
    await h.context.runBatchVoiceGeneration('', {}, {}, {}, {designEngine: 'gemini', shouldPause: () => true});
    assert.equal(h.context.bulkVoiceGenerationState.next_index, 1);
    h.context.speakerReadyState.actor = true;
    await h.context.runBatchVoiceGeneration('', {}, {}, {}, {designEngine: 'gemini', resume: true, includeReady: true});
    assert.equal(h.context.bulkVoiceGenerationState.include_ready, false, 'resume uses the original override');
    assert.equal(h.processed.length, 0);
    assert.equal(h.context.bulkVoiceGenerationState.skipped_count, 2);
    await h.context.runBatchVoiceGeneration('', {}, {}, {}, {designEngine: 'gemini', includeReady: true});
    assert.deepEqual(h.processed.map(v => v.speaker), ['ready', 'actor']);
    assert.ok(h.processed.every(v => v.options.candidateCount === 2 && v.options.includeReady));
});

test('Auto assign excludes ready speakers in proposals and rechecks them at apply time', () => {
    const h = setup();
    h.fields['batch-include-ready'] = {checked: false};
    h.context.availableChatterboxVoices = [{name: 'ready', prompt_path: 'new-ready.wav'}, {name: 'actor', prompt_path: 'new-actor.wav'}];
    h.context.turboSelectionState = {ready: 'approved.wav', actor: 'old.wav'};
    h.context.autoAssignTableBody = {querySelectorAll: () => [
        {dataset: {speaker: 'ready'}, value: 'new-ready.wav'},
        {dataset: {speaker: 'actor'}, value: 'new-actor.wav'}]};
    h.context.document.querySelectorAll = () => [];
    h.context.closeAutoAssignModal = () => {};
    load(h.context, 'function getAutoAssignVoiceChoices(', 'function shouldSkipReadySpeaker(');
    load(h.context, '    function fuzzyMatchScore(', '    function renderAutoAssignTable(');
    load(h.context, '    async function applyAutoAssignSelections(', '    if (autoAssignThresholdSlider)');
    assert.deepEqual(Array.from(h.context.buildAutoAssignProposals(), p => p.speaker), ['actor']);
    h.context.speakerReadyState.actor = true;
    h.context.applyAutoAssignSelections();
    assert.deepEqual(h.context.turboSelectionState, {ready: 'approved.wav', actor: 'old.wav'});
    h.fields['batch-include-ready'].checked = true;
    assert.equal(h.context.buildAutoAssignProposals().length, 2);
    h.context.applyAutoAssignSelections();
    assert.deepEqual(h.context.turboSelectionState, {ready: 'new-ready.wav', actor: 'new-actor.wav'});
});

test('local batch helper retains pending candidates and ready speakers without provider calls', async () => {
    const h = setup();
    const candidates = [{id: 'existing', prompt_path: 'existing.wav'}];
    h.context.speakerVoiceDesignCandidates = {actor: candidates};
    load(h.context, 'async function generateSpeakerVoicePromptBatch(', 'async function generateGeminiSpeakerBatch(');
    const before = JSON.stringify(h.profiles);
    const skipped = await h.context.generateSpeakerVoicePromptBatch('ready', '', null, 'qwen3');
    const reused = await h.context.generateSpeakerVoicePromptBatch('actor', '', null, 'qwen3', {candidateCount: 4});
    assert.equal(skipped.skipped, true);
    assert.equal(reused.reused, true);
    assert.equal(h.context.speakerVoiceDesignCandidates.actor, candidates);
    assert.equal(JSON.stringify(h.profiles), before);
    assert.equal(h.requests.length, 0);
});
