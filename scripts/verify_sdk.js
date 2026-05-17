
import { Langvision } from './dist/index.js';

async function main() {
    console.log('Initializing Langvision client...');
    const client = new Langvision({ cliPath: 'echo' }); // Mock CLI with echo

    console.log('Testing finetune...');
    // Should print the command args via echo
    await client.finetune({
        model: 'gpt2',
        dataset: 'data.txt',
        epochs: 1,
        batchSize: 2,
        learningRate: 1e-4,
        loraRank: 8,
        outputDir: './test-output'
    });

    console.log('\nTesting generate...');
    const output = await client.generate('gpt2', {
        prompt: 'Hello world',
        maxTokens: 10
    });
    console.log('Generate output:', output);
}

main().catch(console.error);
