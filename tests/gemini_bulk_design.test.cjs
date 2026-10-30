const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const main = fs.readFileSync(path.join(__dirname, '../static/js/main.js'), 'utf8');
const gemini = fs.readFileSync(path.join(__dirname, '../static/js/gemini-tts.js'), 'utf8');
function setup(fail = false) {
    const profiles = {narrator: {voice: 'Warm and clear', voice_candidate_count: 2}, actor: {voice_design_prompt: 'Squeaky', voice_candidate_count: 4}};
    const calls = [], storage = new Map(), assigned = [];
    const context = vm.createContext({window: {geminiTtsVoices: [], isGeminiTtsEngine: engine => engine?.startsWith('gemini_tts')},
        getSelectedJobEngine: () => 'gemini_tts', speakerProfiles: profiles, speakerReadyState: {},
        assignGeminiSpeakerVoice: async (...args) => {assigned.push(args); return true;},
        activeProjectId: 'book', bulkVoiceGenerationState: {gemini_model: 'gemini_tts_lite'},
        findSpeakerProfile: name => ({profile: profiles[name]}), normalizeSpeakerKey: v => v,
        updateSpeakerProfileEntry: (name, values) => Object.assign(profiles[name], values),
        localStorage: {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value)},
        fetch: async (url, options) => {
            const request = JSON.parse(options.body); calls.push(request); const id = 'voice_' + calls.length;
            const ok = !(fail && request.name.endsWith('Candidate B'));
            return {ok, json: async () => ({success: ok, id, name: request.name, error: 'rate limit'})};
        }});
    vm.runInContext(gemini.slice(gemini.indexOf('const geminiAuditionStorageKey'), gemini.indexOf('// Kept separate from local sample creation')), context);
    vm.runInContext(main.slice(main.indexOf('async function generateGeminiSpeakerBatch('), main.indexOf('function bulkVoiceSpeakerSignature(')), context);
    return {context, profiles, calls, storage, assigned};
}
test('Gemini model controls are hidden and disabled for local engines on initial load and switches', () => {
    const engine = {value: 'qwen3', addEventListener: (_, fn) => {engine.change = fn;}};
    const options = {}, model = {}, count = {value: '8'};
    const elements = {'speaker-batch-gemini-options': options, 'speaker-batch-gemini-model': model, 'speaker-batch-candidate-count': count};
    const context = vm.createContext({batchDesignEngineSelect: engine, document: {getElementById: id => elements[id]}});
    const start = main.indexOf('    const updateBatchGeminiControls =');
    vm.runInContext(main.slice(start, main.indexOf('    const batchStatus', start)), context);
    assert.equal(options.hidden, true); assert.equal(model.disabled, true);
    engine.value = 'gemini'; engine.change();
    assert.equal(options.hidden, false); assert.equal(model.disabled, false); assert.equal(count.max, '4');
    engine.value = 'breeze'; engine.change();
    assert.equal(options.hidden, true); assert.equal(model.disabled, true); assert.equal(count.max, '10');
    const html = fs.readFileSync(path.join(__dirname, '../templates/index.html'), 'utf8');
    assert.match(html, /id="speaker-batch-gemini-options" hidden/);
});
test('bulk Gemini creates isolated hosted auditions using the requested model and per-speaker description', async () => {
    const {context, profiles, calls, storage} = setup();
    await context.generateGeminiSpeakerBatch('narrator', 'Book narrator', {});
    await context.generateGeminiSpeakerBatch('actor', 'Book actor', {});
    assert.equal(calls.length, 6);
    assert.ok(calls.every(v => v.engine === 'gemini_tts_lite'));
    assert.equal(calls[0].description, 'Warm and clear');
    assert.equal(calls[2].description, 'Squeaky');
    assert.equal(profiles.narrator.gemini_voice_audition.candidates.length, 2);
    assert.equal(profiles.actor.gemini_voice_audition.candidates.length, 4);
    assert.equal(profiles.actor.voice_design_engine, 'gemini');
    assert.equal(profiles.actor.gemini_voice_audition.selected_id, '');
    assert.notEqual(profiles.actor.gemini_voice_audition.group_id, profiles.narrator.gemini_voice_audition.group_id);
    assert.ok([...storage.keys()].some(key => key.startsWith('ttsStory.geminiPending')));
    await context.generateGeminiSpeakerBatch('actor', 'Book actor', {});
    assert.equal(calls.length, 6, 'resume/start-over must not rebill an existing audition');
});
test('partial Gemini bulk failures retain completed candidates and never call local save APIs', async () => {
    const {context, profiles, calls} = setup(true);
    await assert.rejects(context.generateGeminiSpeakerBatch('narrator', 'Narrator', {}), /1\/2 candidates saved/);
    assert.equal(profiles.narrator.gemini_voice_audition.candidates.length, 1);
    assert.match(profiles.narrator.gemini_voice_audition.last_error, /rate limit/);
    assert.equal(calls.length, 2);
    await context.generateGeminiSpeakerBatch('narrator', 'Narrator', {});
    assert.equal(calls.length, 2);
});

test('a single bulk Gemini voice is assigned to its own speaker, including prefixed names', async () => {
    const {context, profiles, calls, assigned} = setup();
    profiles.narrator.voice_candidate_count = profiles.actor.voice_candidate_count = 1;
    assert.equal((await context.generateGeminiSpeakerBatch('narrator', 'test narrator', {})).assigned, true);
    assert.equal((await context.generateGeminiSpeakerBatch('actor', 'test actor', {})).assigned, true);
    assert.deepEqual(assigned.map(v => v.slice(0,3)), [['narrator','voice_1','test narrator'],['actor','voice_2','test actor']]);
    assert.equal(profiles.narrator.gemini_voice_audition.selected_id, 'voice_1');
    assert.equal(profiles.actor.gemini_voice_audition.selected_id, 'voice_2');
    assert.equal(calls.length, 2);
});

test('single Gemini bulk results replace the preset in the actual per-speaker dropdowns', async () => {
    const h=setup();
    const selects=['narrator','actor'].map(speaker=>({dataset:{speaker},value:'Iapetus',options:[{value:'Iapetus'}],
        add(option){this.options.push(option);},dispatchEvent(){}}));
    h.context.document={querySelectorAll:()=>selects};
    h.context.Option=function(text,value){this.textContent=text;this.value=value;};
    h.context.Event=function(type){this.type=type;};
    const start=main.indexOf('async function assignGeminiSpeakerVoice(');
    vm.runInContext(main.slice(start,main.indexOf('function renderSpeakerProfileSummary(',start)),h.context);
    h.profiles.narrator.voice_candidate_count=h.profiles.actor.voice_candidate_count=1;
    await h.context.generateGeminiSpeakerBatch('narrator','test narrator',{});
    await h.context.generateGeminiSpeakerBatch('actor','test actor',{});
    assert.deepEqual(selects.map(v=>v.value),['voice_1','voice_2']);
    assert.equal(selects[0].options.at(-1).textContent,'test narrator');
    assert.equal(selects[1].options.at(-1).textContent,'test actor');
    assert.equal(h.profiles.narrator.selected_voice_id,'voice_1');
    assert.equal(h.profiles.actor.selected_voice_name,'test actor');
    assert.equal(h.profiles.actor.selected_voice_path,'');
});

test('an older unselected single Gemini audition can be assigned without billing again', async () => {
    const {context, profiles, calls, assigned} = setup();
    profiles.actor.voice_candidate_count = 1;
    profiles.actor.gemini_voice_audition = {engine:'gemini_tts', candidates:[{id:'voice_saved', name:'test actor'}], selected_id:''};
    const result = await context.generateGeminiSpeakerBatch('actor', 'test actor', {});
    assert.equal(result.reused, true);
    assert.equal(result.assigned, true);
    assert.equal(calls.length, 0);
    assert.equal(assigned[0][1], 'voice_saved');
});

test('an assignment failure keeps the created single voice available for a no-cost retry', async () => {
    const h=setup();
    h.profiles.narrator.voice_candidate_count=1;
    h.context.assignGeminiSpeakerVoice=async()=>{throw new Error('Dropdown unavailable');};
    await assert.rejects(h.context.generateGeminiSpeakerBatch('narrator','test narrator',{}), /Dropdown unavailable/);
    assert.equal(h.calls.length,1);
    assert.equal(h.profiles.narrator.gemini_voice_audition.candidates.length,1);
    assert.equal(h.profiles.narrator.gemini_voice_audition.selected_id,'');
    h.context.assignGeminiSpeakerVoice=async()=>true;
    const result=await h.context.generateGeminiSpeakerBatch('narrator','test narrator',{});
    assert.equal(result.assigned,true);
    assert.equal(h.calls.length,1);
});

test('multiple candidates, partial failures, ready speakers, and non-Gemini narration are never auto-picked', async () => {
    for (const mode of ['multiple', 'partial', 'ready', 'local']) {
        const h=setup(mode === 'partial');
        h.profiles.narrator.voice_candidate_count = mode === 'multiple' || mode === 'partial' ? 2 : 1;
        if(mode === 'ready') h.context.fetch = async (...args) => { h.context.speakerReadyState.narrator=true; return {ok:true,json:async()=>({success:true,id:'voice_ready',name:'Ready'})}; };
        if(mode === 'local') h.context.getSelectedJobEngine=()=> 'breeze_tts_2';
        try {await h.context.generateGeminiSpeakerBatch('narrator', 'Narrator', {});} catch(error) {assert.equal(mode, 'partial');}
        assert.equal(h.assigned.length, 0, mode);
        assert.equal(h.profiles.narrator.gemini_voice_audition.selected_id, '', mode);
    }
});

test('Auto assign offers custom Gemini IDs and exact speaker auditions, never presets or local samples', async () => {
    const h=setup();
    h.profiles.narrator.voice_candidate_count=1;
    h.profiles.narrator.gemini_voice_audition={engine:'gemini_tts',requested_count:1,candidates:[{id:'voice_narrator',name:'test narrator'}]};
    h.profiles.actor.gemini_voice_audition={engine:'gemini_tts',candidates:[{id:'voice_a',name:'test actor A'},{id:'voice_b',name:'test actor B'}]};
    h.context.window.geminiTtsVoices=[{short_name:'Iapetus',display_name:'Iapetus'},{short_name:'voice_other',display_name:'Designed: Other',designed:true}];
    h.context.availableChatterboxVoices=[{name:'narrator',prompt_path:'local.wav'}];
    h.context.currentStats={speakers:['narrator','actor']};
    h.context.includeReadyInBatchActions=()=>false;
    h.context.document={querySelectorAll:()=>[]};
    h.context.showNotification=()=>{};
    h.context.autoAssignTableBody={querySelectorAll:()=>[
        {dataset:{speaker:'narrator'},value:'voice_narrator'}, {dataset:{speaker:'actor'},value:''}]};
    h.context.closeAutoAssignModal=()=>{};
    const start=main.indexOf('    function fuzzyMatchScore(');
    vm.runInContext(main.slice(start, main.indexOf('    function renderAutoAssignTable(',start)), h.context);
    const apply=main.indexOf('    async function applyAutoAssignSelections(');
    vm.runInContext(main.slice(apply, main.indexOf('    if (autoAssignThresholdSlider)',apply)), h.context);
    const voices=h.context.getAutoAssignVoiceChoices();
    assert.ok(voices.every(v=>v.prompt_path.startsWith('voice_')));
    const proposals=h.context.buildAutoAssignProposals();
    assert.equal(proposals[0].bestPromptValue,'voice_narrator');
    assert.equal(proposals[0].score,1);
    assert.equal(proposals[1].needsReview,true);
    await h.context.applyAutoAssignSelections();
    assert.deepEqual(h.assigned.map(v=>v.slice(0,2)),[['narrator','voice_narrator']]);
    assert.equal(h.profiles.narrator.gemini_voice_audition.selected_id,'voice_narrator');
    assert.ok(!h.profiles.actor.gemini_voice_audition.selected_id);
    assert.equal(h.calls.length,0);
});

test('the next cloud speaker starts while the preceding checkpoint is still saving', async () => {
    const processed = [];
    let release;
    const context = vm.createContext({currentStats: {speakers: ['first', 'second']}, bulkVoiceGenerationState: null,
        window: {loadGeminiTtsCatalog: async () => ({success: true})},
        document: {getElementById: () => ({value: '2'})}, hasResumableBulkVoiceGeneration: () => false,
        normalizeVoiceDesignEngine: v => v, shouldSkipReadySpeaker: () => false, bulkVoiceSpeakerSignature: v => v.join('|'), buildBatchVoiceName: (_, v) => v,
        persistBulkVoiceGenerationState: () => context.bulkVoiceGenerationState.next_index === 1
            ? new Promise(resolve => {release = resolve;}) : Promise.resolve(),
        generateSpeakerVoicePromptBatch: async speaker => {processed.push(speaker); return {success: true};},
        getVoiceDesignEngineConfig: () => ({label: 'Gemini'}), showNotification() {}});
    vm.runInContext(main.slice(main.indexOf('async function runBatchVoiceGeneration('), main.indexOf('function buildBatchVoiceName(')), context);
    const run = context.runBatchVoiceGeneration('', {}, {}, {}, {designEngine: 'gemini'});
    await new Promise(resolve => setImmediate(resolve));
    assert.deepEqual(processed, ['first', 'second']);
    assert.equal(typeof release, 'function');
    release();
    await run;
    assert.equal(context.bulkVoiceGenerationState.status, 'completed');
});
test('Gemini preflight checks the dedicated cloud key rather than local engine health', async () => {
    let data = {success: true, design_available: true};
    const context = vm.createContext({normalizeVoiceDesignEngine: v => v, window: {loadGeminiTtsCatalog: async () => data}});
    const start = main.indexOf('async function requireVoiceDesignBackend(');
    const end = main.indexOf('    const engineConfig', start);
    vm.runInContext(main.slice(start, end) + '\n}', context);
    await context.requireVoiceDesignBackend('gemini');
    data = {success: true, design_available: false};
    await assert.rejects(context.requireVoiceDesignBackend('gemini'), /dedicated Gemini TTS API key/);
});

test('bulk pause/resume checkpoints speakers and preserves the Gemini model without local voice refreshes', async () => {
    const processed = [], checkpoints = [];
    const context = vm.createContext({currentStats: {speakers: ['narrator', 'actor']}, bulkVoiceGenerationState: null,
        window: {loadGeminiTtsCatalog: async () => ({success: true})},
        document: {getElementById: id => ({value: id === 'speaker-batch-gemini-model' ? 'gemini_tts_lite' : '2'})},
        hasResumableBulkVoiceGeneration: () => context.bulkVoiceGenerationState?.status === 'paused',
        normalizeVoiceDesignEngine: v => v, shouldSkipReadySpeaker: () => false, bulkVoiceSpeakerSignature: v => v.join('|'),
        buildBatchVoiceName: (_, speaker) => speaker,
        persistBulkVoiceGenerationState: async () => checkpoints.push(JSON.parse(JSON.stringify(context.bulkVoiceGenerationState))),
        generateSpeakerVoicePromptBatch: async (speaker, _, __, engine) => {assert.equal(engine, 'gemini'); processed.push(speaker); return {success: true};},
        getVoiceDesignEngineConfig: () => ({label: 'Gemini'}), showNotification() {},
        refreshChatterboxVoices: () => {throw Error('Cloud casting must not use local sample refresh');}});
    vm.runInContext(main.slice(main.indexOf('async function runBatchVoiceGeneration('), main.indexOf('function buildBatchVoiceName(')), context);
    const paused = await context.runBatchVoiceGeneration('', {}, {}, {}, {designEngine: 'gemini', shouldPause: () => true});
    assert.equal(paused.paused, true);
    assert.equal(context.bulkVoiceGenerationState.next_index, 1);
    assert.equal(context.bulkVoiceGenerationState.gemini_model, 'gemini_tts_lite');
    context.document.getElementById = () => ({value: 'gemini_tts'});
    await context.runBatchVoiceGeneration('', {}, {}, {}, {designEngine: 'gemini', resume: true});
    assert.deepEqual(processed, ['narrator', 'actor']);
    assert.equal(context.bulkVoiceGenerationState.gemini_model, 'gemini_tts_lite');
    assert.equal(checkpoints.at(-1).status, 'completed');
});
