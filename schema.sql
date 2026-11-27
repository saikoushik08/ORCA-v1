-- ORCA database migration. Run once in the Supabase SQL editor.
-- Safe to run repeatedly. The backend still works without these columns (it falls
-- back to status-only updates) but then cannot show stage, progress or failure reasons.

-- jobs: schema updates
alter table jobs
  add column if not exists preset         text default 'standard',
  add column if not exists current_stage  text,
  add column if not exists progress       int default 0,
  add column if not exists error_message  text,
  add column if not exists error_code     text,
  add column if not exists error_stage    text,
  add column if not exists attempt        int default 1,
  add column if not exists started_at     timestamptz,
  add column if not exists finished_at    timestamptz,
  add column if not exists updated_at     timestamptz,
  add column if not exists heartbeat_at   timestamptz,
  add column if not exists owner_id       uuid,
  add column if not exists parent_job_id  uuid,
  add column if not exists result         jsonb,
  add column if not exists input_type     text default 'photos'; -- 'photos' | 'video'

-- images: schema updates
alter table images
  add column if not exists original_name  text,
  add column if not exists sha256         text,
  add column if not exists width          int,
  add column if not exists height         int,
  add column if not exists size_bytes     bigint,
  add column if not exists camera_key     text,
  add column if not exists cloud_synced   boolean default false,
  add column if not exists source_type    text default 'photo', -- 'photo' | 'video_frame'
  add column if not exists source_video   text,
  add column if not exists frame_index    int,
  add column if not exists frame_time_s   real;

create index if not exists images_job_id_idx on images (job_id);
create index if not exists jobs_status_created_idx on jobs (status, created_at);

-- Audit trail: one row per state change.
create table if not exists job_events (
  id          bigserial primary key,
  job_id      uuid not null,
  ts          timestamptz default now(),
  event_type  text,
  from_status text,
  to_status   text,
  stage       text,
  message     text,
  data        jsonb
);
create index if not exists job_events_job_idx on job_events (job_id, id);

-- Atomic queue claim (PostgreSQL SKIP LOCKED)
create or replace function claim_next_job() returns setof jobs language sql as $$
  update jobs
  set status = 'processing',
      started_at = coalesce(started_at, now()),
      updated_at = now(),
      heartbeat_at = now()
  where job_id = (
    select job_id from jobs
    where status = 'pending'
    order by created_at
    limit 1
    for update skip locked
  )
  returning *;
$$;
