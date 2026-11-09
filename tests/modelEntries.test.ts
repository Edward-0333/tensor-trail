import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { validateModel } from '@tensor-trail/model-core/agent';
import { parseProject, serializeProject } from '@tensor-trail/model-core/core/serialization';
import { clone } from '@tensor-trail/model-core/core/types';
import { registry } from '@tensor-trail/model-core/modules';
import { example } from '@tensor-trail/model-core/modules/examples';

const root = fileURLToPath(new URL('../', import.meta.url));

describe('共享源码与独立 Node SDK 的旧项目兼容', () => {
  it.each(['mixed', 'pluto-input'] as const)('%s 项目经 JSON 往返后两种入口一致', kind => {
    const project = example(kind);
    project.hyperparameters = { history_steps: { type: 'int', value: 21 } };
    if (kind === 'mixed') {
      const input = project.graph.nodes[0];
      input.tensorColor = '#123456';
      const alias = clone(input);
      alias.id = 'input_alias';
      alias.valueSource = input.id;
      alias.position.x += 100;
      project.graph.nodes.push(alias);
      project.graph.edges[0].waypoints = [{ x: 80, y: -20 }, { x: 240, y: -20 }];
      project.graph.edges[0].labelPosition = 0.3;
    }
    const parsed = parseProject(serializeProject(project), registry);
    const sourceResult = validateModel(parsed);
    // The input-only PLUTO example intentionally lacks a final Output node.
    expect(sourceResult.ok).toBe(kind === 'mixed');
    const cli = spawnSync(process.execPath, ['packages/cli/scripts/agent.mjs', 'validate', '-'], {
      cwd: root, input: serializeProject(parsed), encoding: 'utf8', windowsHide: true,
    });
    expect(cli.status, cli.stderr).toBe(kind === 'mixed' ? 0 : 1);
    const bundledResult = JSON.parse(cli.stdout);
    expect(bundledResult.ok).toBe(sourceResult.ok);
    expect(bundledResult.diagnostics).toEqual(sourceResult.diagnostics);
    expect(bundledResult.outputs).toEqual(sourceResult.outputs);
    expect(bundledResult.project).toEqual(sourceResult.project);
    expect(bundledResult.project.hyperparameters).toEqual(project.hyperparameters);
    if (kind === 'mixed') {
      expect(bundledResult.project.graph.nodes.at(-1).valueSource).toBe('input');
      expect(bundledResult.project.graph.edges[0]).toMatchObject({
        waypoints: project.graph.edges[0].waypoints, labelPosition: 0.3,
      });
    }
  });
});
