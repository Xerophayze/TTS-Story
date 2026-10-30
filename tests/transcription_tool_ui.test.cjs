const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

test('a ready transcription utility never enables synthesis or appears as a TTS engine', () => {
    const source = fs.readFileSync(path.join(__dirname, '../static/js/settings.js'), 'utf8');
    const context = vm.createContext({engineSetupCatalog: [
        {id: 'voice_transcription', kind: 'utility', ready: true},
        {id: 'kokoro', kind: 'local', ready: false},
    ]});
    vm.runInContext(source.slice(source.indexOf('function readyEngineIds('), source.indexOf('function filterEngineSelectors(')), context);
    assert.equal(context.readyEngineIds().size, 0);
    context.engineSetupCatalog[1].ready = true;
    assert.deepEqual(Array.from(context.readyEngineIds()), ['kokoro']);
    const template = fs.readFileSync(path.join(__dirname, '../templates/index.html'), 'utf8');
    assert.match(template, /id="utility-tabs-label">Utilities/);
    assert.match(template, /id="engine-panel-transcription"/);
    assert.doesNotMatch(template, /<option[^>]+value="voice_transcription"/);
});
