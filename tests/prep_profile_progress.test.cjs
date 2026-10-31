const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

test('Prep text shows the final profile stage until the profile request completes', async () => {
    const source = fs.readFileSync('static/js/main.js', 'utf8');
    const run = source.slice(source.indexOf('async function _runGeminiPrep('), source.indexOf('function updateGeminiProgress('));
    const progress = [];
    let finishProfiles;
    const profiles = new Promise(resolve => { finishProfiles = resolve; });
    const input = {value: ''}, pause = {}, abort = {style: {}};
    const button = {textContent: 'Prep text', disabled: false};
    const context = {
        document: {getElementById: id => ({'input-text': input, 'gemini-pause-btn': pause, 'gemini-abort-btn': abort})[id] || null},
        getEnabledSectionHeadings: () => [], getSelectedGeminiPromptOverride: () => '',
        updateGeminiProgress: state => progress.push(state), _hidePrepResumePanel() {},
        showNotification() {}, resolveBookTitleFromSections: () => '', latestGeminiBookTitle: '',
        _clearPrepProgress: async () => {}, analyzeText: async () => true,
        fetchSpeakerProfiles: () => profiles, console,
        _geminiPrepPauseRequested: false, _geminiPrepAbortRequested: false, lastAnalyzedText: '',
    };
    vm.runInNewContext(run, context);
    const completion = context._runGeminiPrep(button, 'original', 'hash', {
        sections: [{title: 'Chapter 1'}], outputs: ['[narrator]Hello.[/narrator]'], known_speakers: []
    });
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(progress.at(-1).label, 'Preparing character profiles and voice types…');
    assert.equal(progress.at(-1).count, 'Final step');
    assert.equal(progress.at(-1).indeterminate, true);
    assert.equal(button.disabled, true);
    assert.equal(pause.disabled, true);
    finishProfiles();
    await completion;
    assert.equal(progress.at(-1).visible, false);
    assert.equal(button.disabled, false);
});
