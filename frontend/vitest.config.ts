import { defineConfig, configDefaults } from "vitest/config";

export default defineConfig({
  esbuild: {
    jsx: "automatic",
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./test/setup.ts"],
    globals: true,
    clearMocks: true,
    exclude: [...configDefaults.exclude, "e2e/**"],
  },
});
