CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    predicted INTEGER NOT NULL,
    confidence REAL NOT NULL,
    true_label INTEGER
);
