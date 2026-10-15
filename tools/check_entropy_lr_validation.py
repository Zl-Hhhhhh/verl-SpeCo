"""CPU preflight of entropy LR cohorts against the installed verl/data contract."""
import argparse
import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl_speco.trainer.entropy_lr_runtime import EntropyLRRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dataset = RLHFDataset(args.data, tokenizer, OmegaConf.create(dict(
        max_prompt_length=512, filter_overlong_prompts=True,
        filter_overlong_prompts_workers=1, shuffle=False)))
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn, shuffle=False)
    with TemporaryDirectory(prefix='entropy-lr-preflight-') as directory:
        trainer = SimpleNamespace(
            val_dataloader=loader, tokenizer=tokenizer,
            config=SimpleNamespace(trainer=OmegaConf.create({'default_local_dir': directory})),
            _speco_drafter_training_config=lambda: dict(candidate_gate_enabled=True,
                                                       lr_scheduler_type='constant'),
            _speco_resume_global_step_hint=lambda: None)
        trainer._get_gen_batch = lambda batch: RayPPOTrainer._get_gen_batch(trainer, batch)
        runtime = EntropyLRRuntime(trainer)
        runtime.prepare_validation_cohort()
        assert not hasattr(trainer, 'global_steps')
        trainer.global_steps = 5  # Upstream initializes this only inside fit().
        small, large = runtime.validation_batch(32), runtime.validation_batch(64)
        assert small.meta_info['global_steps'] == 5
        trainer.global_steps = 10
        assert runtime.validation_batch(32).meta_info['global_steps'] == 10
        assert small.meta_info['global_steps'] == 5
        assert 2 <= len(small) <= 32 and len(small) <= len(large) <= 64
        assert small.non_tensor_batch['uid'].tolist() == large.non_tensor_batch['uid'][:len(small)].tolist()
        assert len(set(large.non_tensor_batch['uid'])) == len(large)
        assert all(key in large.non_tensor_batch for key in ('raw_prompt', 'reward_model', 'data_source'))
        assert len({json.dumps(prompt, sort_keys=True) for prompt in large.non_tensor_batch['raw_prompt']}) == len(large)
        assert large.meta_info['validate'] and not large.meta_info['do_sample']
        assert large.meta_info['speco_candidate_gate_max_tokens'] == 512
        print(json.dumps(dict(status='passed', dataset_rows=len(dataset),
                              probe_sizes=[len(small), len(large)],
                              generation_fields=sorted(large.non_tensor_batch.keys()))))


if __name__ == '__main__':
    main()
