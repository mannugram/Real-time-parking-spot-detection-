CREATE TABLE parking_status (
    lot_id SERIAL PRIMARY KEY,
    lot_name VARCHAR(50),
    empty_slots INT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

