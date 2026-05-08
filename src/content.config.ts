import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const reviews = defineCollection({
	loader: glob({ base: './src/content/reviews', pattern: '**/*.{md,mdx}' }),
	schema: () =>
		z.object({
			title: z.string(),
			date: z.coerce.date(),
			updated: z.coerce.date().optional(),
			category: z.string(),
			tags: z.array(z.string()).default([]),
			thumbnail: z.string().optional(),
			affiliate_links: z
				.object({
					gap_official: z.string().optional(),
					rakuten: z.string().optional(),
					amazon: z.string().optional(),
				})
				.optional(),
			rating: z.number().min(0).max(5).optional(),
			description: z.string().optional(),
		}),
});

const guides = defineCollection({
	loader: glob({ base: './src/content/guides', pattern: '**/*.{md,mdx}' }),
	schema: () =>
		z.object({
			title: z.string(),
			date: z.coerce.date(),
			updated: z.coerce.date().optional(),
			category: z.string(),
			tags: z.array(z.string()).default([]),
			thumbnail: z.string().optional(),
			description: z.string().optional(),
		}),
});

export const collections = { reviews, guides };
