# scripts/step3Controller_cartpole_model_selection.py
# Controls Step 3 (selecting the champion hyperparameters) by coordinating progress
# across two parallel workers.
# Includes candidate pruning + adaptive early stopping.

import sys
import json
import wandb
import subprocess
import numpy as np

from pathlib import Path
from statistics import mean

from src.utils.wandb_login import login_wandb
from src.utils.logger_setup import setup_logger
from src.utils.load_yaml_config import load_yaml_config
from src.utils.main_common_setup import create_run_and_get_config

logger = setup_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1] # resolve() improves robustness for path resolution
WORKER = PROJECT_ROOT / "scripts/step3Worker_cartpole_model_selection.py"
TOP10_CANDIDATES_YAML = "configs/top10_candidates.yaml" # The worker's start directory may vary (cwd issues), so pass this explicitly.
                                                    # We keep it relative because the worker calls load_yaml_config(), which expects
                                                    # a path like "configs/top10_candidates.yaml" and will join with project root internally.

WANDB_PROJECT = "RL_Project_Data" # Constant: valid project name
WANDB_ENTITY = "foxmir-stanford-university" # Constant: valid entity name
WANDB_JOB_TYPE = "step3Controller" # Constant: default job type tag

def main():
    # 1️⃣ Log in to W&B
    if not login_wandb(): # Combine action + check: login_wandb() returns False on failure.
        logger.critical("!!! [External-1]: W&B login failed; controller exiting !!!")
        sys.exit(1) # 1 indicates abnormal exit (0 indicates success)

    # 2️⃣ Load config file (kept simple; not using CLI args here)
    try:
        default_config = load_yaml_config(config_path="configs/top10_candidates.yaml") # Pass config path; errors are handled inside
    except Exception as e:
        logger.error(f"Error while loading hyperparameter config file: '{e}'", exc_info=True)
        logger.critical("!!! [External-2]: Hyperparameter config load failed; controller exiting !!!")
        sys.exit(1)

    # 3️⃣ Create W&B run + create artifact + upload nested YAML as an artifact for archiving
    logger.info("Preparing to create W&B run + artifact, and upload the nested YAML for archiving...")
    try: # W&B config expects a flat runnable dict; ours is nested, so we init manually and upload YAML as an artifact.
        run = wandb.init(
            project=WANDB_PROJECT,       # 使用在本模块定义的常量
            entity=WANDB_ENTITY,         # 使用在本模块定义的常量
            name=WANDB_JOB_TYPE,     # 使用传入或生成的运行名称，通常是从默认配置文件中传入
            job_type=WANDB_JOB_TYPE,      # 标签区分训练或者分析，方便快速筛选
        )
        logger.info(f"W&B run '{run.name}' created successfully...")
        configs_artifact = wandb.Artifact(name="top10_candidates", type="candidates") 
        configs_artifact.add_file( PROJECT_ROOT / "configs/top10_candidates.yaml") 
        run.log_artifact(configs_artifact) # 发送到云端工件
        logger.info("Step 3 config file uploaded to W&B as an artifact...")

    except Exception as e:
        logger.error(f"Error while creating W&B run/artifact and uploading YAML: '{e}'", exc_info=True)
        logger.critical("!!! [External-3]: Failed to create W&B run/artifact or upload YAML; controller exiting !!!")
        wandb.finish() # Safe to call even if init failed
        sys.exit(1)


    # 4️⃣ Hard-coded seed lists + read initial candidate list from local top10 YAML (top-level keys)
    try:
        independent_train_seeds = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110] # Independent seeds for extra validation of the champion
        train_seeds = [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                31, 32, 33, 34, 35, 36, 37, 38, 39, 40
            ]
        # independent_train_seeds = [101, 102] # 🧨🧨🧨 For quick smoke test
        # train_seeds = [1, 2, 3, 4] # 🧨🧨🧨 For quick smoke test
        
        active_candidates_id_list = list(default_config.keys()) # Active (surviving) candidates; initially all top-level keys (string IDs)
        logger.info(f"Step 3 training seeds: {train_seeds}")
        logger.info(f"Initial candidate list (all active): {active_candidates_id_list}")
    except Exception as e:
        logger.error(f"Error while reading original candidates from sweeps config: '{e}'", exc_info=True)
        logger.critical("!!! [External-4]: Failed to read original candidates; controller exiting !!!")
        wandb.finish() # Attempt to finish run
        sys.exit(1)

    # 5️⃣ The controller manages two independent subprocesses to complete all training + evaluation
    big_dict = {} # Stores per-candidate mean scores across seeds. Example: {"c01": [230,220,400...], "c02": [420,433,200,...], ...}
    for train_seed in train_seeds:
        n_resamples=10000 # Number of bootstrap resamples for 99% paired-difference CI
        np.random.seed(train_seed) # Seeded resampling for reproducibility

        need = active_candidates_id_list.copy() # Copy current active list for this seed; must be a copy since we mutate need
        logger.info(f"🔔🔔🔔 Seed {train_seed}: current active candidates: {need} 🔔🔔🔔")
        # 【5.1】 Run two parallel subprocesses and generate JSON outputs
        while need: # Without early stop, we keep at least 2 candidates, but need can be odd-sized (3, 5, ...), so handle both cases.
            candidate_id_2 = None # 🔺If need becomes empty before step 【5.1.2】, candidate_id_2 would not exist; set a safe default.
            try: # 【5.1.1】 Pop the first candidate + start worker and wait
                candidate_id_1 = need.pop(0)
                p1, out1, log1 = start_worker(candidate_id_1, train_seed) # Start worker for candidate_id_1

                try:
                    if need: # 【5.1.2】 Pop the next candidate and run it (cannot use 'is not None' because [] is not None)
                        candidate_id_2 = need.pop(0)
                        p2, out2, log2 = start_worker(candidate_id_2, train_seed)
                        rc2 = p2.wait()
                        log2.close()
                        if rc2 != 0: # 🔍 Return-code check: non-zero means failure
                            logger.error(f"Seed {train_seed}: the 2nd subprocess failed, return code: {rc2}. Check the subprocess log.")
                            logger.critical("!!![Internal-2]: Failed while running the 2nd subprocess; controller exiting !!!")
                            wandb.finish()
                            sys.exit(1)
                except Exception as e:
                    logger.error(f"Seed {train_seed}: error while running the 2nd worker: '{e}'", exc_info=True)
                    logger.critical("!!![Internal-2]: Failed while running the 2nd worker; exiting !!!")
                    wandb.finish()
                    sys.exit(1)
                
                rc1 = p1.wait() # rc = return code; 0 means success, non-zero indicates failure
                log1.close()
                if rc1 != 0: # 🔍 Return-code check
                    # Note: we wait for p2 first, then p1. If p1 fails early while p2 is still running,
                    # we only discover it after p2 completes.
                    logger.error(f"Seed {train_seed}: the 1st subprocess failed, return code: {rc1}. Check the subprocess log.")
                    logger.critical("!!![Internal-2]: Failed while running the 1st subprocess; controller exiting !!!")
                    wandb.finish()
                    sys.exit(1)


                # 【5.2】 After both workers finish successfully for this seed, read the generated JSON outputs
                try:
                    c_mean_1 = json.loads(out1.read_text(encoding="utf-8"))["c_mean"]
                    big_dict.setdefault(candidate_id_1, []).append(c_mean_1) # Create list on first sight, then append
                    if candidate_id_2 is not None:
                        c_mean_2 = json.loads(out2.read_text(encoding="utf-8"))["c_mean"]
                        big_dict.setdefault(candidate_id_2, []).append(c_mean_2)

                except Exception as e:
                    logger.error(f"Seed {train_seed}: error while reading mean values from worker JSON outputs: '{e}'", exc_info=True)
                    logger.critical("!!![Internal-3]: Failed to read mean values from worker JSON outputs; exiting !!!")
                    wandb.finish()
                    sys.exit(1)

            except Exception as e:
                logger.error(f"Seed {train_seed}: error while running the 1st worker: '{e}'", exc_info=True)
                logger.critical("!!![Internal-1]: Failed while running the 1st worker; exiting !!!")
                wandb.finish()
                sys.exit(1)


        # 📑 Protocol: when the 99% paired-difference CI threshold is 10, apply adaptive stopping and pruning
        if train_seed > 20: # Start from the 21st seed: ranking + stop check + pruning + update active list
        # if train_seed > 2: # 🧨🧨🧨 For quick smoke test: start judging from the 3rd seed
            #📑【5.3】Rank all active candidates by mean over seeds
            candidate_means_over_seeds_dict = {}
            sorted_candidates_name_list = []
            try:
                for n in active_candidates_id_list: # Compute per-candidate mean across seeds
                    candidate_means_over_seeds_dict[n] = mean(big_dict[n])
                sorted_candidates_list = sorted(candidate_means_over_seeds_dict.items(), key=lambda item: item[1], reverse=True) # Sort by mean desc
                sorted_candidates_name_list = [c_id for c_id, _ in sorted_candidates_list] # Extract candidate IDs only
            except Exception as e:
                logger.error(f"Seed {train_seed}: error while ranking candidates by mean over seeds (step 5.3): '{e}'", exc_info=True)
                logger.critical("!!![Judge-1]: Failed to rank candidates by mean over seeds; exiting !!!")
                wandb.finish()
                sys.exit(1)

            # 📑【5.4】Compute 99% paired-difference CI for top1 vs top2, and apply adaptive stopping
            try:
                top1_id = sorted_candidates_name_list[0]
                top2_id = sorted_candidates_name_list[1]

                top1_means_list = big_dict[top1_id]
                top2_means_list = big_dict[top2_id]

                index_matrix = np.random.randint(0, train_seed, size=(n_resamples, train_seed)) # NOTE: this is an index matrix, not a value matrix.
                                                                                                 # Indexing with it is equivalent to bootstrap-resampling
                                                                                                 # 'train_seed' elements 'n_resamples' times. We generate
                                                                                                 # it once outside compute_99_paired_difference_ci() to avoid
                                                                                                 # regenerating it repeatedly for efficiency.

                # Compute paired-difference CI
                ci_low, ci_high = compute_99_paired_difference_ci(top1_means_list, top2_means_list, index_matrix)

                if ci_low > 10: # Adaptive stop decision
                    logger.info(f"""🎉🎉🎉【5.4】【Adaptive Early Stop】From seed {train_seed}, the 99% paired-difference CI between
                                top1 '{top1_id}' and top2 '{top2_id}' is ({ci_low:.2f}, {ci_high:.2f}). The lower bound > 10,
                                so the stopping condition is met and training stops.
                                All collected data: {big_dict}
                                🏆 Champion: {top1_id}
                                Step3_controller finished. 🎉🎉🎉
                                """)
                    break # Break out of the seed loop

                elif len(sorted_candidates_name_list) != 2: # 【5.5】Candidate pruning (if top3+ exist and no early stop)
                    try:
                        for n in sorted_candidates_name_list[2:]: # All candidates except top1/top2 (already compared above)
                            competitor_means_list = big_dict[n] # Candidate's mean list
                            ci_low, ci_high = compute_99_paired_difference_ci(top1_means_list, competitor_means_list, index_matrix)

                            if ci_low > 10: # Pruning decision
                                logger.info(f"""💥【5.5】【Candidate Pruning】From seed {train_seed}, the 99% paired-difference CI between
                                top1 '{top1_id}' and competitor '{n}' is ({ci_low:.2f}, {ci_high:.2f}). The lower bound > 10,
                                so '{n}' is pruned. 💥""")
                                active_candidates_id_list.remove(n) # Remove from active list
                    except Exception as e:
                        logger.error(f"Seed {train_seed}: error during candidate pruning (step 5.5): '{e}'", exc_info=True)
                        logger.critical("!!![Judge-3]: Candidate pruning failed; exiting !!!")
                        wandb.finish()
                        sys.exit(1)

                else: # No early stop and no extra candidates to prune; continue to next seed
                    pass

            except Exception as e:
                logger.error(f"Seed {train_seed}: error during adaptive early-stop check (step 5.4): '{e}'", exc_info=True)
                logger.critical("!!![Judge-2]: Adaptive early-stop check failed; exiting !!!")
                wandb.finish()
                sys.exit(1)

    if train_seed == 40:
        logger.info(f"""🎉🎉🎉 All {len(train_seeds)} seeds finished with no early stop.
                    All collected data: {big_dict}
                    🏆 Champion: {top1_id}
                    Step3_controller finished. 🎉🎉🎉
                    """)

    # 6️⃣ Extra independent validation for the champion
    try:
        logger.info(f"🔔 Running extra independent validation for champion candidate '{top1_id}'! 🔔")
        independent_train_seeds_means = []
        independent_train_seeds_sds = []
        for independent_train_seed in independent_train_seeds[::2]: # Take every 2nd seed to reduce waiting time while keeping CPU busy
            p1, out1, log1 = start_worker(top1_id, independent_train_seed)
            p2, out2, log2 = start_worker(top1_id, independent_train_seed + 1)
            rc1 = p1.wait()
            rc2 = p2.wait()
            log1.close()
            log2.close()
            if rc1 != 0 or rc2 != 0: # 0 means success; if either fails, exit
                logger.error(f"!!! During independent validation, subprocess 1 or 2 failed (return codes: {rc1}, {rc2}). Check logs; exiting !!!")
                sys.exit(1)

            for w in [out1,out2]: # Order does not matter
                json_data = json.loads(w.read_text(encoding="utf-8"))
                independent_train_seeds_means.append(json_data["c_mean"])
                independent_train_seeds_sds.append(json_data["c_sd"])

        means_array = np.array(independent_train_seeds_means, dtype=np.float64)
        independent_mean = means_array.mean()
        independent_sd = means_array.std(ddof=1)
        logger.info(f"🔔🔔🔔 Independent evaluation for champion '{top1_id}': mean={independent_mean:.2f}, sd={independent_sd:.2f} 🔔🔔🔔")
    except Exception as e:
        logger.error(f"Error during extra independent validation for the champion: '{e}'", exc_info=True)
        logger.critical("!!![External-6]: Champion independent validation failed; exiting !!!")
        try:
            log1.close()
            log2.close()
        except:
            pass
        wandb.finish()
        sys.exit(1)


    # 7️⃣ Upload results: save big_dict JSON as an artifact + update W&B summary
    try:
        big_dict_path = PROJECT_ROOT / "data/step3/big_dict.json"
        big_dict_path.parent.mkdir(parents=True, exist_ok=True)
        big_dict_path.write_text(
            json.dumps(big_dict, indent=2), # indent keeps the JSON readable
            encoding="utf-8")

        big_dict_artifact = wandb.Artifact(name="step3_big_dict", type="results")
        big_dict_artifact.add_file(str(big_dict_path))
        wandb.run.log_artifact(big_dict_artifact)
        logger.info("Step 3 big_dict JSON has been uploaded to W&B as an artifact...")

        wandb.run.summary.update({
            "stop_seed": train_seed,
            "top1_id": top1_id,
            "top1_mean_over_seed": mean(top1_means_list),
            "top1_independent_mean": independent_mean,
            "top1_independent_sd": independent_sd
        })
    except Exception as e:
        logger.error(f"Error while uploading data: '{e}'", exc_info=True)
        logger.critical("!!![External-7]: Data upload failed; exiting !!!")
        wandb.finish()
        sys.exit(1)

    wandb.finish()

    logger.info("Step3_controller completed successfully! Enjoy Step4 sensitivity analysis! 🎉🎉🎉")

# 🍎 Subprocess launcher
def start_worker(candidate_id: str, train_seed: int):
    try:
        out_path = PROJECT_ROOT / "data/step3" / f"output_{candidate_id}_seed{train_seed}.json" # The controller defines the JSON path.
        out_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"step3_worker_{candidate_id}_seed{train_seed}.log"
        log_f = open(log_path, "w", encoding="utf-8", newline="")  

        cmd = [
            sys.executable, str(WORKER),
            "--top10_candidates_yaml", str(TOP10_CANDIDATES_YAML),
            "--candidate_id", candidate_id, # 当然我们的candidate_id本身就是字符串
            "--main_seed", str(train_seed),
            "--out", str(out_path)
        ]

        p = subprocess.Popen(cmd, stdout = log_f, stderr = subprocess.STDOUT) # Start an independent subprocess; pipe logs to a dedicated file
    except Exception as e:
        logger.error(f"Seed {train_seed}: error while starting subprocess for candidate {candidate_id}: '{e}'", exc_info=True)
        logger.critical("!!![Function-1]: Subprocess launcher failed; exiting !!!")
        try:
            log_f.close()
        except:
            pass
        raise
    return p, out_path , log_f # Return subprocess object + output file path


# 🍎 Compute 99% paired-difference confidence interval under bootstrap resampling
def compute_99_paired_difference_ci(top1_means_list, competitor_means_list, index_matrix, confidence_level=0.99):
    try:
        differences_list = np.array(top1_means_list) - np.array(competitor_means_list) # Pairwise differences
        resampled_means = differences_list[index_matrix].mean(axis=1) # Map indices to values, then take row-wise means

        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        
        ci_low = np.percentile(resampled_means, lower_percentile)
        ci_high = np.percentile(resampled_means, upper_percentile)
    except Exception as e:
            logger.error(f"Error while computing 99% paired-difference CI via bootstrap: '{e}'", exc_info=True)
            logger.critical("!!![Function-2]: Bootstrap CI computation failed; exiting !!!")
            raise
    return ci_low, ci_high


# Standard entry point
if __name__ == "__main__":
    main()