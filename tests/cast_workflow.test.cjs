const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');
const source = fs.readFileSync(path.join(__dirname, '../static/js/main.js'), 'utf8');

function loadFunction(ctx, signature) {
    const start = source.indexOf(signature);
    assert.ok(start >= 0, signature);
    const next = /\n(?:async )?function /g;
    next.lastIndex = start + signature.length;
    const end = next.exec(source)?.index ?? source.length;
    vm.runInContext(source.slice(start, end), ctx);
}

test('generation preflight catches a partially assigned cast and opens its first missing speaker', async () => {
    const notices = [], opened = [];
    const context = vm.createContext({document: {getElementById: () => ({value: 'Manuscript'})},
        currentStats: {has_speaker_tags: true, speakers: ['Alice', 'Bob']}, lastAnalyzedText: 'Manuscript', _tagIssues: [],
        _updateTagErrorBanner() {}, getVoiceAssignments: () => ({Alice: {voice: 'alba'}}),
        showNotification: message => notices.push(message), openSpeakerEditModal: name => opened.push(name),
        fetch: () => {throw new Error('Preflight must not submit generation');}});
    loadFunction(context, 'function findUnassignedSpeakers(');
    loadFunction(context, 'async function submitGenerateAudioJob(');
    await context.submitGenerateAudioJob();
    assert.deepEqual(opened, ['Bob']);
    assert.match(notices[0], /Bob/);
    assert.deepEqual(Array.from(context.findUnassignedSpeakers({default: {}}, ['Alice', 'Bob'])), []);
    assert.deepEqual(Array.from(context.findUnassignedSpeakers({Alice: {}, Bob: {}}, ['Alice', 'Bob'])), []);
});

test('plain-text preflight checks the visible Speaker 1, not the analyzer default pseudo-speaker', async () => {
    const context = vm.createContext({document: {getElementById: () => ({value: 'Manuscript'})},
        currentStats: {has_speaker_tags: false, speakers: ['default']}, lastAnalyzedText: 'Manuscript', _tagIssues: [],
        _updateTagErrorBanner() {}, getVoiceAssignments: () => ({'Speaker 1': {voice: 'alba'}}),
        showNotification() {throw new Error('Must not report a missing voice');},
        openSpeakerEditModal() {throw new Error('Must not open a nonexistent default speaker');},
        console: {log() {throw new Error('Preflight accepted');}}});
    loadFunction(context, 'function findUnassignedSpeakers(');
    loadFunction(context, 'async function submitGenerateAudioJob(');
    await assert.rejects(context.submitGenerateAudioJob(), /Preflight accepted/);
});

test('Cast rendering indexes assignment rows once and prioritizes the edited drawer row', () => {
    let reads = 0;
    const names = Array.from({length: 250}, (_, i) => `actor-${i}`);
    const hidden = names.map(name => ({dataset: {speaker: name}}));
    const edited = {dataset: {speaker: 'actor-0'}, edited: true};
    const seen = [];
    const context = vm.createContext({window: {}, document: {readyState: 'loading', addEventListener() {},
        querySelectorAll: () => [edited]}, currentStats: {speakers: names},
        getAssignmentRows: () => {reads++; return hidden;}});
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/cast-view.js'), 'utf8'), context);
    context.names = names;
    vm.runInContext('castSpeakerNames = names;', context);
    context.castVoiceInfo = (name, row) => {seen.push(row); return {primary: name, state: 'set'};};
    const entries = context.castEntries();
    assert.equal(entries.length, 250);
    assert.equal(reads, 1);
    assert.equal(seen[0], edited);
    assert.equal(seen[249], hidden[249]);
    const ready = {name: 'ready', ready: true, voice: {state: 'set'}};
    const review = {name: 'review', ready: false, voice: {state: 'set'}};
    const missing = {name: 'missing', ready: true, voice: {state: 'missing'}};
    assert.equal(context.castNextToReview([ready, review, missing]), missing);
    assert.equal(context.castNextToReview([ready, review]), review);
    assert.equal(context.castNextToReview([ready]), null);
});

test('undated legacy projects do not prevent newly saved projects sorting first', () => {
    const context = vm.createContext({});
    loadFunction(context, 'function getProjectTimestamp(');
    const projects = [{name: 'missing'}, {name: 'old', saved_at: '2025-01-01'},
        {name: 'invalid', saved_at: 'not a date'}, {name: 'new', saved_at: '2026-10-29'},
        {name: 'fallback', saved_at: 'bad', created_at: '2024-01-01'}];
    projects.sort((a, b) => context.getProjectTimestamp(b) - context.getProjectTimestamp(a));
    assert.deepEqual(projects.map(p => p.name), ['new', 'old', 'fallback', 'missing', 'invalid']);
});

test('Library data is not fetched while the Generate page is active', async () => {
    const library = fs.readFileSync(path.join(__dirname, '../static/js/library.js'), 'utf8');
    let active = false, requests = 0, renders = 0;
    const context = vm.createContext({document: {getElementById: id => id === 'library-tab' ? {classList: {contains: () => active}} : null},
        fetch: async () => {requests++; return {json: async () => ({success: true, items: []})};},
        displayLibraryItems: () => {renders++;}, libraryLastListingSignature: '', chunkReviewModalJobId: null});
    vm.runInContext(library.slice(library.indexOf('async function loadLibrary('), library.indexOf('// Display library items')), context);
    await context.loadLibrary();
    assert.equal(requests, 0);
    active = true;
    await context.loadLibrary();
    assert.equal(requests, 1);
    assert.equal(renders, 1);
    active = false;
    await context.loadLibrary({force: true});
    assert.equal(requests, 2, 'an explicitly forced refresh is still allowed');
});
