-- ==========================
-- BigQuery AI Demo Templates
-- ==========================
-- Fill these once per session:
DECLARE PROJECT_ID STRING DEFAULT 'your-project-id';
DECLARE DATASET    STRING DEFAULT 'your_dataset';
DECLARE BUCKET     STRING DEFAULT 'your-gcs-bucket';  -- e.g., gs://accessible-mvp-bucket

-- 1) OPTIONAL: Create dataset if it doesn’t exist
EXECUTE IMMEDIATE FORMAT("""
  CREATE SCHEMA IF NOT EXISTS `%s.%s`
""", PROJECT_ID, DATASET);

-- 2) OPTIONAL: Create an Object Table over frames stored in GCS
--    Upload local frames (mvp_outputs/vision/frames/*.jpg) to
--    gs://<BUCKET>/frames/scene_xxx_*.jpg first.
-- NOTE: This uses Object Tables. If you use Connections, adapt to your environment.
EXECUTE IMMEDIATE FORMAT("""
  CREATE OR REPLACE EXTERNAL TABLE `%s.%s.frames`
  WITH PARTITION COLUMNS (_PARTITIONDATE DATE)
  OPTIONS (
    uris = ['%s/frames/*'],
    object_metadata = 'SIMPLE'
  )
""", PROJECT_ID, DATASET, BUCKET);

-- 3) OPTIONAL: Load BLIP captions (CSV) to BigQuery for AI processing
--    Upload your CSV to GCS (e.g., gs://<BUCKET>/scene_vision.csv) then:
EXECUTE IMMEDIATE FORMAT("""
  CREATE OR REPLACE EXTERNAL TABLE `%s.%s.scene_vision_csv`
  OPTIONS (
    format = 'CSV',
    uris   = ['%s/scene_vision.csv'],
    skip_leading_rows = 1,
    autodetect = true
  )
""", PROJECT_ID, DATASET, BUCKET);

-- 4) Example: Use AI.GENERATE to summarize per-scene captions into a concise hint
-- NOTE: Depending on your environment, use AI.GENERATE or AI.GENERATE_TEXT with a MODEL option.
-- Replace MODEL with an available model in your region, e.g. 'gemini-1.5-flash'.
-- The result column is named 'content'.
CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET}.scene_hints` AS
SELECT
  scene_id,
  AI.GENERATE(
    MODEL => 'gemini-1.5-flash',
    PROMPT => '''
You are generating very short, vivid audio description hints for blind users.
Summarize the following frame captions into one concise sentence (max 18 words),
objective, no spoilers, and use present continuous tense.

Captions:
''' || STRING_AGG(caption, '\n'),
    PARAMS => STRUCT(0.2 AS temperature)
  ) AS hint
FROM (
  SELECT
    REGEXP_EXTRACT(object_name, r'scene_(\d+)') AS scene_id,
    caption
  FROM `${PROJECT_ID}.${DATASET}.scene_vision_csv`
)
GROUP BY scene_id;

-- 5) Example: Create embeddings for captions and build a simple vector search
-- If you have BigQuery remote models configured, you can call ML.GENERATE_EMBEDDING.
-- Replace the MODEL below with an embedding model you’ve registered (e.g., text-embedding-004).
CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET}.caption_embeddings` AS
SELECT
  object_name,
  caption,
  ML.GENERATE_EMBEDDING(
    MODEL `${PROJECT_ID}.${DATASET}.embedding_model`,
    caption
  ) AS embedding
FROM `${PROJECT_ID}.${DATASET}.scene_vision_csv`;

-- 6) Example: VECTOR_SEARCH (without index) – find similar captions to a query
-- Replace the embedding array function below with the correct constructor if needed.
WITH q AS (
  SELECT ML.GENERATE_EMBEDDING(
    MODEL `${PROJECT_ID}.${DATASET}.embedding_model`,
    'woman enters lab, white coat, serious expression'
  ) AS qvec
)
SELECT
  object_name,
  caption,
  VECTOR_SEARCH(distance_type => 'COSINE', query => (SELECT qvec FROM q), candidates => embedding) AS score
FROM `${PROJECT_ID}.${DATASET}.caption_embeddings`
ORDER BY score ASC
LIMIT 10;

-- 7) Example: AI.GENERATE_TABLE to convert raw rows into a structured table of scene descriptors
CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET}.structured_scene_descriptors` AS
SELECT *
FROM AI.GENERATE_TABLE(
  MODEL => 'gemini-1.5-flash',
  PROMPT => '''
Given scene captions and time spans, generate JSON with keys:
scene_id (string), setting (short), main_entities (array of strings), tone (one word), hint (max 18 words).
Be neutral and spoiler-free.
''',
  INPUT_TABLE => (
    SELECT
      REGEXP_EXTRACT(object_name, r'scene_(\d+)') AS scene_id,
      caption
    FROM `${PROJECT_ID}.${DATASET}.scene_vision_csv`
  ),
  JSON_SCHEMA => '''
{
  "scene_id": "STRING",
  "setting": "STRING",
  "main_entities": "ARRAY<STRING>",
  "tone": "STRING",
  "hint": "STRING"
}
'''
);
