import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

const require = createRequire(import.meta.url);
const { resolveNpmCli } = require('../scripts/npm-toolchain.cjs');
const {
  desktopEntry,
  launcherPath,
  installLauncher,
  desktopId,
} = require('../scripts/linux-launcher.cjs');
let temporary: string;
beforeEach(() => {
  temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'tt-linux-'));
});
afterEach(() => {
  fs.rmSync(temporary, { recursive: true, force: true });
});
function file(relative: string, content = '') {
  const target = path.join(temporary, relative);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, content);
  return target;
}

describe('跨平台 Node/npm 定位', () => {
  it.skipIf(process.platform === 'win32')(
    'Linux Node 压缩包和 Conda 符号链接找到配套 lib/node_modules/npm',
    () => {
      const node = file('runtime/bin/node');
      const npm = file('runtime/lib/node_modules/npm/bin/npm-cli.js');
      const link = path.join(temporary, 'conda-node');
      fs.symlinkSync(node, link, 'file');
      expect(resolveNpmCli(link, {})).toBe(npm);
    },
  );
  it('Windows Node 布局仍支持 node.exe 旁的 npm', () => {
    const node = file('windows/node.exe');
    const npm = file('windows/node_modules/npm/bin/npm-cli.js');
    expect(resolveNpmCli(node, {})).toBe(npm);
  });
  it('Ubuntu 系统布局支持 share/nodejs，不把不存在的 npm_execpath 当成成功', () => {
    const node = file('usr/bin/node');
    const npm = file('usr/share/nodejs/npm/bin/npm-cli.js');
    expect(resolveNpmCli(node, { npm_execpath: '/not/installed/npm-cli.js' })).toBe(npm);
  });
  it('没有 npm 时返回明确错误而非执行任意同名 shell 文件', () => {
    const node = file('bin/node');
    file('bin/npm', '#!/bin/sh\nexit 0');
    expect(() => resolveNpmCli(node, { PATH: path.dirname(node) })).toThrow('npm CLI not found');
  });
});

it('启动器按锁文件更新已有安装，安装失败后下次启动重试', () => {
  const entry = file('scripts/ensure-dependencies.cjs',
    fs.readFileSync(require.resolve('../scripts/ensure-dependencies.cjs'), 'utf8'));
  const lock = file('package-lock.json', 'first lock');
  const installed = file('node_modules/.package-lock.json');
  const calls = file('calls', '');
  const npm = file('scripts/npm.cjs', `
    const fs = require('node:fs');
    fs.appendFileSync('calls', process.argv.slice(2).join(' ') + '\\n');
    if (process.argv.includes('--ignore-scripts')) {
      const log = fs.openSync('contender-error', 'w');
      const contender = require('node:child_process').spawnSync(process.execPath,
        ['scripts/ensure-dependencies.cjs'], { stdio: ['ignore', 'ignore', log] });
      fs.closeSync(log);
      fs.writeFileSync('contender-status', String(contender.status));
    }
    if (fs.existsSync('fail')) process.exit(1);
    fs.writeFileSync('node_modules/.package-lock.json', '{}');
  `);
  const run = (...args: string[]) => spawnSync(process.execPath, [entry, ...args], { encoding: 'utf8' });
  expect(run('--ignore-scripts').status).toBe(0);
  expect(fs.readFileSync(path.join(temporary, 'contender-status'), 'utf8')).toBe('1');
  expect(fs.readFileSync(path.join(temporary, 'contender-error'), 'utf8')).toContain('另一个启动窗口正在安装依赖');
  expect(fs.readFileSync(calls, 'utf8')).toBe('ci --no-audit --no-fund --ignore-scripts\n');
  expect(run().status).toBe(0);
  expect(fs.readFileSync(calls, 'utf8').trim().split('\n')).toHaveLength(1);

  fs.writeFileSync(lock, 'new lock');
  const failure = file('fail');
  expect(run().status).toBe(1);
  fs.unlinkSync(failure);
  expect(run().status).toBe(0);
  fs.unlinkSync(installed);
  expect(run().status).toBe(0);
  expect(fs.readFileSync(calls, 'utf8').trim().split('\n')).toHaveLength(4);
  // A healthy tree should start without invoking npm at all.
  fs.unlinkSync(npm);
  expect(run().status).toBe(0);
});

describe.skipIf(process.platform !== 'linux')('Ubuntu 应用菜单', () => {
  it('尊重绝对 XDG 路径，相对 XDG 值回退到当前用户目录', () => {
    expect(launcherPath({ XDG_DATA_HOME: temporary }, '/user')).toBe(
      path.join(temporary, 'applications', desktopId),
    );
    expect(launcherPath({ XDG_DATA_HOME: 'relative' }, '/user')).toBe(
      path.join('/user/.local/share/applications', desktopId),
    );
  });
  it('安装入口记录实际 Node，不依赖 GNOME 的 PATH 或当前工作目录', () => {
    const node = file('conda/bin/node');
    const root = path.join(temporary, '中文 项目');
    const destination = path.join(temporary, 'applications/test.desktop');
    expect(installLauncher(root, node, destination)).toBe(destination);
    const content = fs.readFileSync(destination, 'utf8');
    expect(content).toContain(`Exec="${node}" "${root}/scripts/linux-desktop.cjs" "--from-menu"`);
    expect(content).toContain(`Path=${root}`);
    expect(content).toContain('Terminal=false');
    expect(content).not.toContain('--no-sandbox');
    expect(content).not.toContain('%f');
  });
  it('路径中的百分号、引号、反斜杠和 shell 字符按 desktop 语法转义', () => {
    const content = desktopEntry('/tmp/percent%"$`\\project', '/opt/node');
    const command = content.split('\n').find((line: string) => line.startsWith('Exec='))!;
    expect(command).toContain('percent%%');
    for (const character of ['"', '$', '`']) expect(command).toContain('\\\\' + character);
    expect(command).toContain('\\\\\\\\project');
    expect(() => desktopEntry('/tmp/a\nb', '/opt/node')).toThrow('换行');
  });
});
