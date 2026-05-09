import { defineCollection, z } from 'astro:content';

const reviews = defineCollection({
	type: 'content',
	schema: z.object({
		title: z.string(),
		description: z.string(),
		date: z.coerce.date(),
		updated: z.coerce.date().optional(),
		category: z.string(),
		tags: z.array(z.string()).default([]),
		thumbnail: z.string().optional(),
		rating: z.number().min(0).max(5).optional(),
		affiliate_links: z.record(z.string()).optional(),
	}),
});

const guides = defineCollection({
	type: 'content',
	schema: z.object({
		title: z.string(),
		description: z.string(),
		date: z.coerce.date(),
		updated: z.coerce.date().optional(),
		category: z.string(),
		tags: z.array(z.string()).default([]),
		thumbnail: z.string().optional(),
	}),
});

const blog = defineCollection({
	type: 'content',
	schema: z.object({
		title: z.string(),
		description: z.string(),
		pubDate: z.coerce.date(),
		updatedDate: z.coerce.date().optional(),
		heroImage: z.string().optional(),
	}),
});

export const collections = { reviews, guides, blog };
