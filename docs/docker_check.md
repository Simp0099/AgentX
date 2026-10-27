# Docker and CI verification output

## Clean clone without a checkpoint

Command:

```text
git clone . /tmp/clean && cd /tmp/clean && docker build -t deepfake-localizer .
```

Clone output:

```text
Cloning into '/tmp/clean'...
done.
```

The clean clone contains the tracked checkpoint placeholder only:

```text
$ git -C /tmp/clean ls-files checkpoints
checkpoints/.gitkeep
```

Build output:

```text
ERROR: failed to connect to the docker API at unix:///Users/bhavishya/.docker/run/docker.sock; check if the path is correct and if the daemon is running: dial unix /Users/bhavishya/.docker/run/docker.sock: connect: no such file or directory
```

## Checkpoint and rebuild

The original repository already has a trained checkpoint:

```text
$ ls -lh checkpoints/best_localizer.keras
-rw-r--r--@ 1 bhavishya staff 234M Oct 24 15:14 checkpoints/best_localizer.keras
```

Copied that checkpoint into the clean clone and retried:

```text
$ docker build -t deepfake-localizer .
ERROR: failed to connect to the docker API at unix:///Users/bhavishya/.docker/run/docker.sock; check if the path is correct and if the daemon is running: dial unix /Users/bhavishya/.docker/run/docker.sock: connect: no such file or directory
```

## Container run

Command: `docker run -p 7860:7860 deepfake-localizer`

```text
failed to connect to the docker API at unix:///Users/bhavishya/.docker/run/docker.sock; check if the path is correct and if the daemon is running: dial unix /Users/bhavishya/.docker/run/docker.sock: connect: no such file or directory
```

**NOT VERIFIED** — the Docker daemon is unavailable, so the image could not be built and the container did not start. First 30 container log lines: **NOT VERIFIED; no container logs were produced.**

## CI workflow check

Command: `rg -n -A2 -B1 'name: Lint|name: Tests|ruff check|pytest tests/' .github/workflows/ci.yml`

```text
24-
25:      - name: Lint
26:        run: ruff check src/ app/ tests/ scripts/ agent/
27-
28:      - name: Tests
29-        # Tests build the model with weights=None, so no ImageNet download and
30-        # no GPU is needed -- the whole suite runs on a free CI runner.
31:        run: pytest tests/ -v
32-        env:
33-          TF_CPP_MIN_LOG_LEVEL: "3"
```
