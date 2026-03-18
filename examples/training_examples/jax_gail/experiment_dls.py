import os
import traceback
from dataclasses import fields

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from loco_mujoco.algorithms import GAILJax
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf
from loco_mujoco.utils import MetricsHandler
from loco_mujoco.utils.metrics import QuantityContainer
from loco_mujoco.environments.humanoids import MpxUnitreeH1


@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment_dls(config: DictConfig):
    """
    Single-script pipeline:
      A) Load a LAFAN1 trajectory into `UnitreeH1` (downloaded/loaded by the existing factory).
      B) Load that same trajectory into `MpxUnitreeH1`.
      C) Create an expert transitions dataset and train GAIL.
    """
    try:
        os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "

        result_dir = HydraConfig.get().runtime.output_dir

        # setup wandb
        wandb.login()
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        run = wandb.init(project=config.wandb.project, config=config_dict)

        # Phase A/B:
        # Load a LAFAN1 trajectory for `UnitreeH1`, then load the same trajectory into `MpxUnitreeH1`
        # (custom XML differs mainly in foot collision/contact geometry).

        # 1) Source trajectory (LAFAN1)
        source_env_name = "UnitreeH1"
        lafan1_group = config.experiment.get("lafan1_dataset_group", "LAFAN1_LOCOMOTION_DATASETS")
        lafan1_conf = LAFAN1DatasetConf(dataset_group=lafan1_group)
        src_env = ImitationFactory.make(source_env_name, lafan1_dataset_conf=lafan1_conf)
        traj_src = src_env.th.traj

        # 2) Target env and load the source trajectory into it
        target_env_params = dict(config.experiment.env_params)
        target_env_params.pop("env_name", None)  # we'll explicitly use MpxUnitreeH1

        env = MpxUnitreeH1(
            init_state_type="TrajInitialStateHandler",
            terminal_state_type="RootPoseTrajTerminalStateHandler",
            terrain_type="FlatFloorTerrain",
            **target_env_params,
        )
        env.load_trajectory(traj=traj_src, warn=False)

        # 3) Build the expert transitions (obs/next_obs/absorbing/dones/rewards)
        expert_dataset = env.create_dataset()

        # Phase C: training (continue with `env` as constructed above)

        agent_conf = GAILJax.init_agent_conf(env, config)
        agent_conf = agent_conf.add_expert_dataset(expert_dataset)

        mh = MetricsHandler(config, env) if config.experiment.validation.active else None

        train_fn = GAILJax.build_train_fn(env, agent_conf, mh=mh)
        train_fn = jax.jit(jax.vmap(train_fn)) if config.experiment.n_seeds > 1 else jax.jit(train_fn)

        rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds + 1)]
        rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
        out = train_fn(_rng)

        agent_state = out["agent_state"]
        save_path = GAILJax.save_agent(result_dir, agent_conf, agent_state)
        run.config.update({"agent_save_path": save_path})

        import time

        t_start = time.time()
        if not config.experiment.debug:
            training_metrics = out["training_metrics"]
            validation_metrics = out["validation_metrics"]

            training_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), training_metrics)
            validation_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), validation_metrics)

            for i in range(len(training_metrics.mean_episode_return)):
                run.log(
                    {
                        "Mean Episode Return": training_metrics.mean_episode_return[i],
                        "Mean Episode Length": training_metrics.mean_episode_length[i],
                        "Discriminator/Output Policy": training_metrics.discriminator_output_policy[i],
                        "Discriminator/Output Expert": training_metrics.discriminator_output_expert[i],
                    },
                    step=int(training_metrics.max_timestep[i]),
                )

                if (i + 1) % config.experiment.validation_interval == 0 and config.experiment.validation.active:
                    run.log(
                        {
                            "Validation Info/Mean Episode Return": validation_metrics.mean_episode_return[i],
                            "Validation Info/Mean Episode Length": validation_metrics.mean_episode_length[i],
                        },
                        step=int(training_metrics.max_timestep[i]),
                    )

                    metrics_to_log = {}
                    for field in fields(validation_metrics):
                        attr = getattr(validation_metrics, field.name)
                        if isinstance(attr, QuantityContainer):
                            measure_name = field.name
                            for field_attr in fields(attr):
                                attr_name = field_attr.name
                                attr_value = getattr(attr, attr_name)
                                if attr_value.size > 0:
                                    metrics_to_log[
                                        f"Validation Measures/{measure_name}/{attr_name}"
                                    ] = attr_value[i]

                    run.log(metrics_to_log, step=int(training_metrics.max_timestep[i]))

                    site_rpos = validation_metrics.euclidean_distance.site_rpos[i]
                    site_rrotvec = validation_metrics.euclidean_distance.site_rrotvec[i]
                    site_rvel = validation_metrics.euclidean_distance.site_rvel[i]
                    run.log(
                        {"Metric for Sweep": site_rpos + site_rrotvec + site_rvel},
                        step=int(training_metrics.max_timestep[i]),
                    )

        print(f"Time taken to log metrics: {time.time() - t_start}s")

        GAILJax.play_policy(
            env,
            agent_conf,
            agent_state,
            deterministic=True,
            n_steps=200,
            n_envs=20,
            record=True,
            train_state_seed=0,
        )
        video_file = env.video_file_path
        run.log({"Agent Video": wandb.Video(video_file)})
        wandb.finish()

    except Exception:
        with open("experiment_dls_crash.log", "w") as f:
            f.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    experiment_dls()

