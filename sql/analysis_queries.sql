-- Run these queries against data/music_analytics.db after:
-- python database.py

-- 1. Dataset overview
SELECT COUNT(*) AS track_count,
       ROUND(AVG(popularity), 2) AS avg_popularity,
       ROUND(AVG(tempo), 1) AS avg_tempo
FROM tracks;

-- 2. Average popularity by release decade
SELECT (release_year / 10) * 10 AS decade,
       COUNT(*) AS tracks,
       ROUND(AVG(popularity), 2) AS avg_popularity
FROM tracks
GROUP BY decade
ORDER BY decade;

-- 3. Popularity by energy bucket
SELECT
  CASE
    WHEN energy < 0.33 THEN 'Low energy'
    WHEN energy < 0.66 THEN 'Medium energy'
    ELSE 'High energy'
  END AS energy_group,
  COUNT(*) AS tracks,
  ROUND(AVG(popularity), 2) AS avg_popularity
FROM tracks
GROUP BY energy_group
ORDER BY avg_popularity DESC;

-- 4. Top artists by average popularity (minimum 5 tracks)
SELECT artist_name, COUNT(*) AS track_count,
       ROUND(AVG(popularity), 2) AS avg_popularity
FROM tracks
GROUP BY artist_name
HAVING COUNT(*) >= 5
ORDER BY avg_popularity DESC
LIMIT 10;

-- 5. Acousticness vs instrumentalness groups
SELECT
  CASE WHEN acousticness >= 0.5 THEN 'More acoustic' ELSE 'Less acoustic' END AS acoustic_group,
  CASE WHEN instrumentalness >= 0.5 THEN 'More instrumental' ELSE 'Less instrumental' END AS instrumental_group,
  COUNT(*) AS tracks,
  ROUND(AVG(popularity), 2) AS avg_popularity
FROM tracks
GROUP BY acoustic_group, instrumental_group
ORDER BY avg_popularity DESC;
