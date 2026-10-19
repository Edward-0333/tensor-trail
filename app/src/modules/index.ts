import { ModuleRegistry } from './registry';
import { primitives } from './primitives';
import { installComposites } from './composites';
export const registry = new ModuleRegistry();
primitives.forEach((def) => registry.register(def));
installComposites(registry);
