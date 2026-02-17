import { z } from 'zod';

export const FinetuneConfigSchema = z.object({
    model: z.string(),
    dataset: z.string(),
    epochs: z.number().optional().default(3),
    batchSize: z.number().optional().default(4),
    learningRate: z.number().optional().default(2e-4),
    loraRank: z.number().optional().default(16),
    outputDir: z.string().optional().default('./output'),
});

export type FinetuneConfig = z.infer<typeof FinetuneConfigSchema>;

export interface GenerationOptions {
    prompt: string;
    maxTokens?: number;
    temperature?: number;
}
