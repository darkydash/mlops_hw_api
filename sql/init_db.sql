CREATE TABLE IF NOT EXISTS error_logs (
    id bigserial primary key,
    error_code int,
    error_message varchar(250) NOT NULL
);