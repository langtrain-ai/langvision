import { execa } from 'execa';
import { FinetuneConfig, FinetuneConfigSchema, GenerationOptions } from './types';

export class Langvision {
    private cliPath: string;
    private apiKey?: string;

    constructor(options: { cliPath?: string, apiKey?: string } = {}) {
        this.cliPath = options.cliPath || 'langvision';
        this.apiKey = options.apiKey;
    }

    private getEnv() {
        return {
            ...process.env,
            LANGVISION_API_KEY: this.apiKey || process.env.LANGVISION_API_KEY
        };
    }

    /**
     * Run a fine-tuning job using the local CLI
     */
    async finetune(config: FinetuneConfig): Promise<void> {
        const validated = FinetuneConfigSchema.parse(config);

        // Construct CLI arguments
        const args = [
            'train',
            '--model', validated.model,
            '--dataset', validated.dataset,
            '--epochs', validated.epochs.toString(),
            '--batch-size', validated.batchSize.toString(),
            '--learning-rate', validated.learningRate.toString(),
            '--lora-rank', validated.loraRank.toString(),
            '--output-dir', validated.outputDir,
        ];

        try {
            await execa(this.cliPath, args, {
                stdio: 'inherit',
                env: this.getEnv()
            });
        } catch (error) {
            throw new Error(`Langvision fine-tuning failed: ${error}`);
        }
    }

    /**
     * Generate text using a model
     */
    async generate(modelPath: string, options: GenerationOptions): Promise<string> {
        const args = [
            'generate',
            '--model', modelPath,
            '--prompt', options.prompt,
        ];

        if (options.maxTokens) {
            args.push('--max-tokens', options.maxTokens.toString());
        }

        if (options.temperature) {
            args.push('--temperature', options.temperature.toString());
        }

        try {
            const { stdout } = await execa(this.cliPath, args, {
                env: this.getEnv()
            });
            return stdout;
        } catch (error) {
            throw new Error(`Langvision generation failed: ${error}`);
        }
    }
}
