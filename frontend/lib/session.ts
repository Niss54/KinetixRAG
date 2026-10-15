const USER_KEY = "kinetix.user_id";
const SESSION_KEY = "kinetix.session_id";

function storedId(storage: Storage, key: string, prefix: string): string {
  const existing = storage.getItem(key);
  if (existing) return existing;
  const created = `${prefix}-${crypto.randomUUID()}`;
  storage.setItem(key, created);
  return created;
}

export function getChatIdentity() {
  return {
    userId: storedId(localStorage, USER_KEY, "operator"),
    sessionId: storedId(sessionStorage, SESSION_KEY, "session"),
  };
}

