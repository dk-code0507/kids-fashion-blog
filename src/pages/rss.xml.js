import { getCollection } from 'astro:content';
import rss from '@astrojs/rss';
import { SITE_DESCRIPTION, SITE_TITLE } from '../consts';

export async function GET(context) {
	const reviews = await getCollection('reviews');
	const guides = await getCollection('guides');
	const allPosts = [
		...reviews.map((post) => ({ ...post.data, link: `/reviews/${post.id}/` })),
		...guides.map((post) => ({ ...post.data, link: `/guides/${post.id}/` })),
	].sort((a, b) => b.date.valueOf() - a.date.valueOf());

	return rss({
		title: SITE_TITLE,
		description: SITE_DESCRIPTION,
		site: context.site,
		items: allPosts.map((post) => ({
			title: post.title,
			pubDate: post.date,
			description: post.description ?? post.title,
			link: post.link,
		})),
	});
}
