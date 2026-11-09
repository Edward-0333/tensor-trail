import { defineConfig } from 'vite';
export default defineConfig({build:{target:'es2022',outDir:'dist',emptyOutDir:true,minify:false,lib:{entry:'src/agent/index.ts',formats:['es'],fileName:()=> 'tensor-trail-agent.js'}}});
