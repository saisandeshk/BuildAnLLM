# Frontend Tests

Frontend tests use Vitest with React Testing Library and JSDOM. Property tests use `fast-check`.

## Install

```bash
npm install
```

## Run

```bash
# One-shot run
npm run test:run

# Watch mode
npm run test
```

## Structure

- `frontend/test/*.test.ts` for lib + hook tests
- `frontend/test/components/*.test.tsx` for React component tests
- `frontend/test/pages/*.test.tsx` for page-level behavior tests

## Notes

- Tests run against JSDOM with mocked browser APIs where needed (EventSource, IntersectionObserver).
- E2E/browser tests are not configured yet.
