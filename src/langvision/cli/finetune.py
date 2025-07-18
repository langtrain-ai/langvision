import argparse
import torch
import logging
import os
import random
import numpy as np
from langvision.models.vision_transformer import VisionTransformer
from langvision.data.datasets import get_dataset
from langvision.training.trainer import Trainer
from langvision.callbacks.early_stopping import EarlyStopping
from langvision.utils.device import get_device
from langvision.utils.cuda import setup_cuda


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for fine-tuning."""
    parser = argparse.ArgumentParser(
        description='Fine-tune VisionTransformer with LoRA and advanced LLM concepts',
        epilog='''\nExamples:\n  langvision finetune --dataset cifar10 --epochs 10\n  langvision finetune --dataset cifar100 --lora_r 8 --rlhf\n''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Data
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Dataset to use')
    data_group.add_argument('--data_dir', type=str, default='./data', help='Dataset directory')
    data_group.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset')
    data_group.add_argument('--img_size', type=int, default=224, help='Input image size (pixels)')
    data_group.add_argument('--patch_size', type=int, default=16, help='Patch size for Vision Transformer')
    data_group.add_argument('--num_workers', type=int, default=2, help='Number of data loader workers')
    # Model
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--embed_dim', type=int, default=768, help='Embedding dimension for ViT')
    model_group.add_argument('--depth', type=int, default=12, help='Number of transformer layers')
    model_group.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    model_group.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP hidden dim ratio')
    model_group.add_argument('--lora_r', type=int, default=4, help='LoRA rank (low-rank adaptation)')
    model_group.add_argument('--lora_alpha', type=float, default=1.0, help='LoRA alpha scaling')
    model_group.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout rate')
    # Training
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    train_group.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    train_group.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    train_group.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'], help='Optimizer type')
    train_group.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (L2 regularization)')
    train_group.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step'], help='Learning rate scheduler')
    train_group.add_argument('--step_size', type=int, default=5, help='StepLR: step size')
    train_group.add_argument('--gamma', type=float, default=0.5, help='StepLR: gamma')
    train_group.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    train_group.add_argument('--eval_only', action='store_true', help='Only run evaluation (no training)')
    # Output
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output_dir', type=str, default='outputs', help='Directory to save outputs and checkpoints')
    output_group.add_argument('--save_name', type=str, default='vit_lora_best.pth', help='Checkpoint file name')
    # Callbacks
    callback_group = parser.add_argument_group('Callbacks')
    callback_group.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    callback_group.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    # CUDA
    cuda_group = parser.add_argument_group('CUDA')
    cuda_group.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    cuda_group.add_argument('--cuda_deterministic', action='store_true', help='Enable deterministic CUDA (reproducible, slower)')
    cuda_group.add_argument('--cuda_benchmark', action='store_true', default=True, help='Enable cudnn.benchmark for fast training')
    cuda_group.add_argument('--cuda_max_split_size_mb', type=int, default=None, help='Set CUDA max split size in MB (for large models, PyTorch >=1.10)')
    # Misc
    misc_group = parser.add_argument_group('Misc')
    misc_group.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    misc_group.add_argument('--log_level', type=str, default='info', help='Logging level (debug, info, warning, error)')
    # Advanced LLM concepts
    llm_group = parser.add_argument_group('Advanced LLM Concepts')
    llm_group.add_argument('--rlhf', action='store_true', help='Use RLHF (Reinforcement Learning from Human Feedback)')
    llm_group.add_argument('--ppo', action='store_true', help='Use PPO (Proximal Policy Optimization)')
    llm_group.add_argument('--dpo', action='store_true', help='Use DPO (Direct Preference Optimization)')
    llm_group.add_argument('--lime', action='store_true', help='Use LIME for model explainability')
    llm_group.add_argument('--shap', action='store_true', help='Use SHAP for model explainability')
    llm_group.add_argument('--cot', action='store_true', help='Use Chain-of-Thought (CoT) prompt generation')
    llm_group.add_argument('--ccot', action='store_true', help='Use Contrastive Chain-of-Thought (CCoT)')
    return parser.parse_args()


def setup_logging(log_level: str) -> None:
    """Set up logging with the specified log level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(level=numeric_level, format='[%(levelname)s] %(message)s')


def main() -> None:
    """Main function for fine-tuning VisionTransformer with LoRA and advanced LLM concepts."""
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("[STEP 1] Loading dataset...")
    setup_cuda(seed=args.seed, deterministic=args.cuda_deterministic, benchmark=args.cuda_benchmark, max_split_size_mb=args.cuda_max_split_size_mb)
    set_seed(args.seed)

    # Data
    try:
        train_dataset = get_dataset(args.dataset, args.data_dir, train=True, img_size=args.img_size)
        val_dataset = get_dataset(args.dataset, args.data_dir, train=False, img_size=args.img_size)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    logger.info("[STEP 2] Initializing model...")
    # Model
    try:
        model = VisionTransformer(
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_chans=3,
            num_classes=args.num_classes,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            lora_config={
                'r': args.lora_r,
                'alpha': args.lora_alpha,
                'dropout': args.lora_dropout,
            },
        ).to(args.device)
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return

    logger.info("[STEP 3] Setting up optimizer and scheduler...")
    # Optimizer
    lora_params = [p for n, p in model.named_parameters() if 'lora' in n and p.requires_grad]
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(lora_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(lora_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if args.device == 'cuda' else None

    logger.info("[STEP 4] Setting up trainer...")
    # Callbacks
    callbacks = []
    if args.early_stopping:
        callbacks.append(EarlyStopping(patience=args.patience))

    # Advanced LLM concept objects
    rlhf = None
    ppo = None
    dpo = None
    lime = None
    shap = None
    cot = None
    ccot = None
    if args.rlhf:
        from langvision.concepts.rlhf import RLHF
        class SimpleRLHF(RLHF):
            def train(self, model, data, feedback_fn, optimizer):
                super().train(model, data, feedback_fn, optimizer)
        rlhf = SimpleRLHF()
    if args.ppo:
        from langvision.concepts.ppo import PPO
        class SimplePPO(PPO):
            def step(self, policy, old_log_probs, states, actions, rewards, optimizer):
                super().step(policy, old_log_probs, states, actions, rewards, optimizer)
        ppo = SimplePPO()
    if args.dpo:
        from langvision.concepts.dpo import DPO
        class SimpleDPO(DPO):
            def optimize_with_preferences(self, model, preferences, optimizer):
                super().optimize_with_preferences(model, preferences, optimizer)
        dpo = SimpleDPO()
    if args.lime:
        from langvision.concepts.lime import LIME
        class SimpleLIME(LIME):
            def explain(self, model, input_data):
                return super().explain(model, input_data)
        lime = SimpleLIME()
    if args.shap:
        from langvision.concepts.shap import SHAP
        class SimpleSHAP(SHAP):
            def explain(self, model, input_data):
                return super().explain(model, input_data)
        shap = SimpleSHAP()
    if args.cot:
        from langvision.concepts.cot import CoT
        class SimpleCoT(CoT):
            def generate_chain(self, prompt):
                return super().generate_chain(prompt)
        cot = SimpleCoT()
    if args.ccot:
        from langvision.concepts.ccot import CCoT
        class SimpleCCoT(CCoT):
            def contrastive_train(self, positive_chains, negative_chains):
                super().contrastive_train(positive_chains, negative_chains)
        ccot = SimpleCCoT()

    logger.info("[STEP 5] (Optional) Loading checkpoint if provided...")
    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        scaler=scaler,
        callbacks=callbacks,
        device=args.device,
        rlhf=rlhf,
        ppo=ppo,
        dpo=dpo,
    )

    # Optionally resume
    start_epoch = 0
    best_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    if args.eval_only:
        logger.info("[STEP 6] Running evaluation only...")
        val_loss, val_acc = trainer.evaluate(val_loader)
        logger.info(f"Eval Loss: {val_loss:.4f}, Eval Acc: {val_acc:.4f}")
        return

    logger.info("[STEP 6] Starting training...")
    # Training with per-epoch progress
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"  [Epoch {epoch+1}/{args.epochs}] Starting epoch...")
        best_acc = trainer.fit(
            train_loader,
            val_loader,
            epochs=epoch+1,
            start_epoch=epoch,
            best_acc=best_acc,
            checkpoint_path=os.path.join(args.output_dir, args.save_name),
        )
        logger.info(f"  [Epoch {epoch+1}/{args.epochs}] Done. Best validation accuracy so far: {best_acc:.4f}")
    logger.info(f"[STEP 7] Training complete. Best validation accuracy: {best_acc:.4f}")

    # Example: Use RLHF, PPO, DPO, LIME, SHAP, CoT, CCoT in training
    if rlhf is not None:
        logger.info("[STEP 8] Applying RLHF...")
        def feedback_fn(output):
            return 1.0 if output.sum().item() > 0 else -1.0
        rlhf.train(model, [torch.randn(3) for _ in range(10)], feedback_fn, optimizer)
    if ppo is not None:
        logger.info("[STEP 9] Applying PPO...")
        import torch
        policy = model
        old_log_probs = torch.zeros(10)
        states = torch.randn(10, 3)
        actions = torch.randint(0, args.num_classes, (10,))
        rewards = torch.randn(10)
        ppo.step(policy, old_log_probs, states, actions, rewards, optimizer)
    if dpo is not None:
        logger.info("[STEP 10] Applying DPO...")
        preferences = [(torch.randn(3), 1.0), (torch.randn(3), -1.0)]
        dpo.optimize_with_preferences(model, preferences, optimizer)
    if lime is not None:
        logger.info("[STEP 11] Running LIME explainability...")
        lime_explanation = lime.explain(model, [[0.5, 1.0, 2.0], [1.0, 2.0, 3.0]])
        logger.info(f"LIME explanation: {lime_explanation}")
    if shap is not None:
        logger.info("[STEP 12] Running SHAP explainability...")
        shap_explanation = shap.explain(model, [[0.5, 1.0, 2.0], [1.0, 2.0, 3.0]])
        logger.info(f"SHAP explanation: {shap_explanation}")
    if cot is not None:
        logger.info("[STEP 13] Generating Chain-of-Thought...")
        chain = cot.generate_chain("What is 2 + 2?")
        logger.info(f"CoT chain: {chain}")
    if ccot is not None:
        logger.info("[STEP 14] Running Contrastive Chain-of-Thought...")
        ccot.contrastive_train([['Step 1: Think', 'Step 2: Solve']], [['Step 1: Guess', 'Step 2: Wrong']])
    logger.info("[COMPLETE] All steps finished.")

if __name__ == '__main__':
    main() 