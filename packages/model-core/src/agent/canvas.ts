import type { Project } from '../core/types';
import { clone } from '../core/types';
import { describeModules, getSchemas } from './catalog';
import { applicationError, buildModel, patchModel, validateModel } from './model';

/** Host adapter keeps the same API usable without React or a browser. */
export function createCanvasApi(host: {
  read: () => Project;
  commit: (project: Project) => void;
  subscribe: (changed: () => void) => () => void;
}) {
  const session = crypto.randomUUID();
  let revision = 0;
  const unsubscribe = host.subscribe(() => {
    revision++;
  });
  const currentRevision = () => `${session}:${revision}`;
  const api = Object.freeze({
    apiVersion: 1 as const,
    describeModules,
    getSchemas,
    buildModel,
    validateModel,
    patchModel,
    getModel: () => ({
      apiVersion: 1 as const,
      revision: currentRevision(),
      project: clone(host.read()),
    }),
    applyModel: (request: {
      project: unknown;
      expectedRevision: string;
      allowWarnings?: boolean;
    }) => {
      const response = validateModel(request?.project);
      if (request?.expectedRevision !== currentRevision()) {
        response.ok = false;
        response.diagnostics.push(
          applicationError(
            'REVISION_CONFLICT',
            '画布已发生变化。请重新 getModel、构建或修改并校验后再应用。',
          ),
        );
      }
      if (
        response.diagnostics.some((d) => d.severity === 'warning') &&
        request?.allowWarnings !== true
      ) {
        response.ok = false;
        response.diagnostics.push(
          applicationError(
            'WARNINGS_REQUIRE_ACK',
            '存在形状警告。检查后传入 allowWarnings: true 才能应用。',
          ),
        );
      }
      if (response.ok && response.project) host.commit(response.project);
      return { ...response, revision: currentRevision() };
    },
  });
  return { api, dispose: unsubscribe };
}
