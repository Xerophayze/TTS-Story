const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/js/settings.js', 'utf8');

test('Claude panel follows primary and backup providers without exposing unrelated tuning', () => {
    const elements = new Map();
    const get = id => {
        if (!elements.has(id)) elements.set(id, {style: {}, value: id === 'llm-provider' ? 'anthropic' : ''});
        return elements.get(id);
    };
    const context = {document: {getElementById: get}, llmBackupProfiles: []};
    vm.runInNewContext(source.slice(source.indexOf('function updateLLMSettingsUI('),
                                   source.indexOf('function createLLMBackupProfile(')), context);
    context.updateLLMSettingsUI();
    assert.equal(get('llm-anthropic-settings').style.display, '');
    assert.equal(get('llm-non-gemini-tuning').style.display, 'none');
    assert.equal(get('llm-local-chunk-group').style.display, '');
    get('llm-provider').value = 'local';
    context.llmBackupProfiles = [{provider: 'anthropic'}];
    context.updateLLMSettingsUI();
    assert.equal(get('llm-anthropic-settings').style.display, '');
    assert.equal(get('llm-non-gemini-tuning').style.display, '');
});

function discoveryContext(models, previous = '') {
    const elements = {
        'anthropic-api-key': {value: 'test-key'},
        'anthropic-workspace-id': {value: 'wrkspc_test'},
        'anthropic-models-status': {textContent: ''},
        'anthropic-model': {value: previous, replaceChildren(...options) { this.options = options; }},
    };
    const calls = [];
    const context = {
        document: {getElementById: id => elements[id]},
        Option: function(text, value) { this.text = text; this.value = value; },
        renderLLMModelSuggestions() {},
        fetch: async (url, options) => {
            calls.push({url, body: JSON.parse(options.body)});
            return {ok: true, json: async () => ({success: true, models})};
        },
    };
    vm.runInNewContext(source.slice(source.indexOf('async function fetchAnthropicModels('),
                                   source.indexOf('async function fetchOpenRouterModels(')), context);
    return {context, elements, calls};
}

test('Claude discovery uses direct endpoint, preserves selected model, and does not save settings', async () => {
    const {context, elements, calls} = discoveryContext(['claude-opus-test', 'claude-sonnet-test'], 'claude-opus-test');
    const button = {disabled: false};
    await context.fetchAnthropicModels(button);
    assert.equal(elements['anthropic-model'].value, 'claude-opus-test');
    assert.equal(button.disabled, false);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, '/api/anthropic/models');
    assert.equal(calls[0].body.workspace_id, 'wrkspc_test');
});

test('First discovery prefers an available Sonnet and empty key makes no request', async () => {
    const {context, elements, calls} = discoveryContext(['claude-opus-test', 'claude-sonnet-test']);
    await context.fetchAnthropicModels({disabled: false});
    assert.equal(elements['anthropic-model'].value, 'claude-sonnet-test');
    elements['anthropic-api-key'].value = '';
    await context.fetchAnthropicModels({disabled: false});
    assert.equal(calls.length, 1);
    assert.match(elements['anthropic-models-status'].textContent, /API key first/);
});
