+++
title = "How to Launch the Blog: Running, Configuration, and Deployment Guide"
date = 2024-05-28T14:00:00+08:00
draft = false
description = "Learn how to start a blog from scratch with this step-by-step guide on running, configuring, and deploying Hugo blogging platform."
summary = "Learn how to start a blog from scratch with this step-by-step guide on running, configuring, and deploying Hugo blogging platform."
tags = ['Blogging', 'Setup-Guides']
showWordCount = true
showEdit = true
editURL = "https://github.com/sylvanding/sylvanding.github.io/tree/main/content"
editAppendPath = true
+++

{{< alert "tag" >}}
**Hello and welcome to my brand-new blog! I'm thrilled to have you here and excited to share the news of its launch. After a lot of planning and designing, I've finally rolled out this platform as a space to share my thoughts, experiences, and knowledge.** :star:
{{< /alert >}}

&nbsp;

# Quick Start

## Installation

### Install Hugo

If you haven't used Hugo before, you will need to [install it onto your local machine](https://gohugo.io/installation/). You can check if it's already installed by running the command `hugo version`.

{{< alert >}}
Make sure you are using **Hugo extended version 0.87.0** or later as the theme takes advantage of some of the latest Hugo features.
{{< /alert >}}

You can find detailed installation instructions for your platform in the [Hugo docs](https://gohugo.io/installation/).

### Clone my repo

```bash
git clone --recurse-submodules https://github.com/sylvanding/sylvanding.github.io.git mywebsite
```

## Running server

```bash
cd mywebsite
hugo serve -D
```

## Creating a new post

```bash
hugo new content posts/your-post-title.md
```

{{< alert >}}
确保在发布内容前设置文章的Front Matter中的`draft`为`false`.
{{< /alert >}}

## Deployment

[![Netlify Status](https://api.netlify.com/api/v1/badges/3d7d4e8d-e53b-4a32-b8d4-9720262d310a/deploy-status)](https://app.netlify.com/sites/sylvanding/deploys)

Deploying Hugo on [Netlify](https://www.netlify.com/) can be a simple and efficient process, especially when integrating with GitHub for continuous deployment.

Before connecting to Netlify, ensure your site’s source code has been pushed to GitHub repo:

```bash
git push origin main
```

Then Netlify tend to build the website automatically according to configuration file `netlify.toml`.

That’s it! Your Hugo blog is now live on Netlify. You can customize it with different themes, add plugins, and optimize your content as desired. Netlify also offers advanced features like custom domains, HTTPS, and continuous deployment, which make your blogging experience secure and efficient.

{{< alert >}}
**In summary:** Never directly edit the theme files. Only make customisations in your Hugo project’s sub-directories, not in the themes directory itself.
{{< /alert >}}

Congo is built to take advantage of all the standard Hugo practices. It is designed to allow all aspects of the theme to be customised and overridden without changing any of the core theme files. This allows for a seamless upgrade experience while giving you total control over the look and feel of your website.

In order to achieve this, you should never manually adjust any of the theme files directly. Whether you install using Hugo modules, as a git submodule or manually include the theme in your `themes/` directory, you should always leave these files intact.

The correct way to adjust any theme behaviour is by overriding files using Hugo's powerful [file lookup order](https://gohugo.io/templates/lookup-order/). In summary, the lookup order ensures any files you include in your project directory will automatically take precedence over any theme files.

For example, if you wanted to override the main article template in Congo, you can simply create your own `layouts/_default/single.html` file and place it in the root of your project. This file will then override the `single.html` from the theme without ever changing the theme itself. This works for any theme files - HTML templates, partials, shortcodes, config files, data, assets, etc.

As long as you follow this simple practice, you will always be able to update the theme (or test different theme versions) without worrying that you will lose any of your custom changes.

## Updating Congo

Git submodules can be updated using the `git` command. Simply execute the following command and the latest version of the theme will be downloaded into your local repository:

```bash
git submodule update --remote --merge
```

Once the submodule has been updated, rebuild your site and check everything works as expected.

## Conclusion

**Thank you for visiting my new blog! I'm excited about the journey ahead and look forward to engaging with a community of tech enthusiasts and developers. Stay tuned for more posts where I'll dive deeper into various technical topics, share development tips, and much more. Don’t forget to subscribe and join the conversation in the comments section below!**