-- Database: music_retrieval

-- DROP DATABASE IF EXISTS music_retrieval;

CREATE DATABASE music_retrieval
    WITH
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.utf8'
    LC_CTYPE = 'en_US.utf8'
    LOCALE_PROVIDER = 'libc'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1
    IS_TEMPLATE = False;

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS music_segments;

CREATE TABLE music_segments (
    id SERIAL PRIMARY KEY,
    file_name VARCHAR(255),
    path TEXT,
    instrument VARCHAR(100),
    segment_id INT,
    
    -- Cấu trúc 3 lớp
    layer1_physical vector(2),
    layer2_perceptual vector(3),
    layer3_identity vector(25)
);

select * from music_segments;