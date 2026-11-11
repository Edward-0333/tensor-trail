#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';

// stdout is always one JSON document. No network, credentials or arbitrary-code execution.
const help = {
  usage: [
    'node packages/cli/scripts/agent.mjs describe [moduleType] [--internal]',
    'node packages/cli/scripts/agent.mjs schema [recipe|patch|validationOptions|patchOptions]',
    'node packages/cli/scripts/agent.mjs build recipe.json [--draft] [--out model.json] [--force]',
    'node packages/cli/scripts/agent.mjs validate recipe-or-project.json [--draft]',
    'node packages/cli/scripts/agent.mjs patch model.json patch.json [--draft] [--layout preserve|auto] [--out next.json] [--force]',
  ],
  notes:
    'Run npm run build:agent first. Use - for one stdin input. Exit codes: 0 valid in the selected mode, 1 model errors, 2 usage/I/O errors. Strict validation is default; --draft permits incomplete wiring, not structural/shape errors. Patches preserve existing geometry by default. Sidecars are not read automatically. --out writes only a project valid in the selected mode; --force allows replacing an existing output file.',
};
function usage(message) {
  throw new Error(message);
}
async function read(file) {
  let bytes;
  if (file === '-') {
    const chunks = [];
    let size = 0;
    for await (const chunk of process.stdin) {
      size += chunk.length;
      if (size > 3_000_000) usage('Input exceeds 3 MB.');
      chunks.push(chunk);
    }
    bytes = Buffer.concat(chunks);
  } else {
    const handle = await fs.open(file, 'r');
    try {
      if ((await handle.stat()).size > 3_000_000) usage('Input exceeds 3 MB.');
      bytes = await handle.readFile();
    } finally {
      await handle.close();
    }
  }
  if (bytes.length > 3_000_000) usage('Input exceeds 3 MB.');
  return JSON.parse(bytes.toString('utf8').replace(/^\uFEFF/, ''));
}
try {
  const args = process.argv.slice(2),
    command = args.shift();
  if (!command || ['help', '--help', '-h'].includes(command)) {
    console.log(JSON.stringify({ apiVersion: 1, ...help }, null, 2));
  } else {
    let out,
      force = false,
      internal = false,
      draft = false,
      layout;
    const positional = [];
    for (let i = 0; i < args.length; i++) {
      if (args[i] === '--out') {
        if (out || !args[i + 1] || args[i + 1].startsWith('--')) usage('Expected one output path.');
        out = args[++i];
      } else if (args[i] === '--force') force = true;
      else if (args[i] === '--draft') draft = true;
      else if (args[i] === '--layout') {
        if (layout || !['preserve', 'auto'].includes(args[i + 1]))
          usage('--layout expects preserve or auto.');
        layout = args[++i];
      } else if (args[i] === '--internal') internal = true;
      else if (args[i].startsWith('--')) usage(`Unknown option ${args[i]}`);
      else positional.push(args[i]);
    }
    if (positional.filter((p) => p === '-').length > 1) usage('Only one input may use stdin.');
    if (out && !['build', 'patch'].includes(command)) usage('--out is only valid for build/patch.');
    if (force && !out) usage('--force requires --out.');
    if (internal && command !== 'describe') usage('--internal is only valid for describe.');
    if (draft && !['build', 'validate', 'patch'].includes(command))
      usage('--draft is only valid for build/validate/patch.');
    if (layout && command !== 'patch') usage('--layout is only valid for patch.');
    const options = { mode: draft ? 'draft' : 'strict' };
    let api;
    try {
      api = await import('../../model-core/dist/tensor-trail-agent.js');
    } catch {
      usage('Agent bundle unavailable. Run npm run build:agent first.');
    }
    let result;
    if (command === 'describe' && positional.length <= 1) {
      result = api.describeModules(internal);
      if (positional.length) {
        result.modules = result.modules.filter((m) => m.type === positional[0]);
        if (!result.modules.length) usage(`Unknown module ${positional[0]}`);
      }
    } else if (command === 'schema' && positional.length <= 1) {
      const schemas = api.getSchemas();
      if (
        positional.length &&
        !['recipe', 'patch', 'validationOptions', 'patchOptions'].includes(positional[0])
      )
        usage('Schema must be recipe, patch, validationOptions or patchOptions.');
      result = positional.length ? schemas[positional[0]] : schemas;
    } else if (command === 'build' && positional.length === 1)
      result = api.buildModel(await read(positional[0]), options);
    else if (command === 'validate' && positional.length === 1)
      result = api.validateModel(await read(positional[0]), options);
    else if (command === 'patch' && positional.length === 2)
      result = api.patchModel(await read(positional[0]), await read(positional[1]), {
        ...options,
        ...(layout ? { layout } : {}),
      });
    else usage('Unknown command or invalid argument count.');
    if (out && result.ok && result.project) {
      await fs.writeFile(out, JSON.stringify(result.project, null, 2) + '\n', {
        flag: force ? 'w' : 'wx',
      });
      result.savedTo = path.resolve(out);
    }
    console.log(JSON.stringify(result, null, 2));
    if (result.ok === false) process.exitCode = 1;
  }
} catch (e) {
  console.log(
    JSON.stringify(
      {
        apiVersion: 1,
        ok: false,
        diagnostics: [{ code: 'CLI_ERROR', severity: 'error', path: [], message: e.message }],
        ...help,
      },
      null,
      2,
    ),
  );
  process.exitCode = 2;
}
