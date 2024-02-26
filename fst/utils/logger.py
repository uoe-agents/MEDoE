from abc import ABC, abstractmethod
from collections import defaultdict
import os.path as osp
import numpy as np
import time
import logging
import wandb
import json
from omegaconf import OmegaConf

class Logger:
    def __init__(self, cfg, task=None, path=None):
        lcfg = cfg.logger
        self.fps = lcfg.video_fps
        self.time_last_commit = 0
        self.summary = lcfg.get("print_summary", None)
        if self.summary is not None:
            self.summary = "SUMMARY:\t" + "\t".join(f"{k} : {v}" for k,v in self.summary)

        self.run = wandb.init(
            project=lcfg.project,
            entity=lcfg.entity,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            tags=lcfg.get("tags", None),
            notes=lcfg.get("notes", None),
            group=lcfg.get("group", None),
            mode=lcfg.get("mode", None),
            reinit=True,
            )
        self.define_metrics()

        self.fps = lcfg.video_fps
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)

    def define_metrics(self):
        wandb.define_metric("Distill/step")
        wandb.define_metric("Train/step")
        wandb.define_metric("Train/ep")
        wandb.define_metric("Distill/*", step_metric="Distill/step")
        wandb.define_metric("Train/*", step_metric="Train/step")
        wandb.define_metric("Return/*", step_metric="Train/step")

    def log(self, ep, step, val):
        msg = (
                f"{np.round(time.time()-self.start_time, 4)}, "
                f"{ep}, "
                f"{step}, "
                f"{val}"
                )
        self.logger.info(msg)

    def wandb_log(self, tag, val, step, commit=False):
        wandb.log({tag: val}, step=step, commit=commit)

    def log_results_dict(self, tag, results_dict, step_metric="_step", commit=False):
        start_step = wandb.run.step + 1
        for step, val in results_dict.items():
            wandb.log({tag: val, step_metric: step}, step=start_step+step, commit=False)
        wandb.log({}, commit=commit)

    def _log_team_return(self, eval_metrics, step, extra={}, commit=False):
        team_returns = sum(eval_metrics.values())
        mean = team_returns.mean()
        std = team_returns.std()
        min = team_returns.min()
        max = team_returns.max()
        team_returns = {
            "Return/team/mean": mean,
            "Return/team/std": std,
            "Return/team/min": min,
            "Return/team/max": max,
            }
        team_returns["Train/step"] = step
        team_returns.update(extra)
        wandb.log(
            team_returns,
            step=step,
            commit=commit,
            )
        if commit:
            self.time_last_commit = time.time()
        return {"mean": mean, "std": std, "min": min, "max": max}

    def _log_individual_return(self, eval_metrics, step, extra={}, commit=False):
        returns = {
            **{f"Return/individual/{agent_id}/mean": np.mean(np.array(rtn))
               for agent_id, rtn in eval_metrics.items()},
            **{f"Return/individual/{agent_id}/std": np.std(np.array(rtn))
               for agent_id, rtn in eval_metrics.items()},
            **{f"Return/individual/{agent_id}/min": np.min(np.array(rtn))
               for agent_id, rtn in eval_metrics.items()},
            **{f"Return/individual/{agent_id}/max": np.max(np.array(rtn))
               for agent_id, rtn in eval_metrics.items()},
            }
        returns["Train/step"] = step
        returns.update(extra)
        wandb.log(returns, step=step, commit=commit)
        if commit:
            self.time_last_commit = time.time()
        clean_returns = {
            agent_id: {
                "mean": np.mean(np.array(rtn)),
                "std": np.std(np.array(rtn)),
                "min": np.min(np.array(rtn)),
                "max": np.max(np.array(rtn)),
                }
            for agent_id, rtn in eval_metrics.items()
            }
        return clean_returns

    def log_eval_metrics(self, eval_metrics, step, extra={}, commit=False):
        team_return = self._log_team_return(eval_metrics, step, extra, commit)
        individual_return = self._log_individual_return(eval_metrics, step, extra, commit)
        return {"team": team_return, "individual": individual_return}

    def log_train_metrics(self, agent_metrics, step, extra={}, commit=False):
        train_metrics = {f"Train/{agent_id}/{metric}": val 
                         for agent_id, metrics in agent_metrics.items()
                         for metric, val in metrics.items()}
        train_metrics["Train/step"] = step
        train_metrics.update(extra)
        wandb.log(train_metrics, step=step, commit=commit)
        if commit:
            self.time_last_commit = time.time()

    def log_video(self, tag, vid, step=None, fps=None, extra={}, commit=False):
        if fps is None:
            fps = self.fps
        wandb.log({
            tag: wandb.Video(vid.squeeze().transpose([0,3,1,2]), fps=fps),
            "Train/step": step
            }, step=step)

    def log_ansi(self, vid, step=None, fps=None):
        # Code adapted from https://github.com/openai/gym/blob/634afec9bc948cc9c2633827fe7af9e67497035e/gym/wrappers/monitoring/video_recorder.py#L267
        if fps is None:
            fps = self.fps
        frame_duration = 1.0/fps
        clear_code = "\u001b[2K\u001b[0;0H"
        events = [
            (frame_duration,
             (clear_code + frame.replace("\n", "\r\n"))
            )
            for frame in vid
            ]
        height = max(frame.count("\n") for frame in vid) + 1
        width = (
            max(max(len(line) for line in frame.split("\n")) for frame in vid)
            + 2
        )
        data = {
            "version": 1,
            "width": width,
            "height": height,
            "duration": len(vid) * frame_duration,
            "command": "-",
            "title": f"Render at timestep {step}",
            "env": {},  # could add some env metadata here
            "stdout": events,
        }
        with open(f"render_{step}.json", "w") as f:
            json.dump(data, f)

    def logmsg(self, msg):
        self.logger.info(msg)

    def print_summary(self):
        if self.summary is not None:
            self.logmsg(self.summary)

    def close(self):
        self.print_summary()
        self.commit()
        self.run.finish()

    def commit(self):
        wandb.log({}, commit=True)
        self.time_last_commit = time.time()
