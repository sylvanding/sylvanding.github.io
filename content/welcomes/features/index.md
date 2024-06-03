+++
title = "Welcome to My New Blog: Powered by Hugo and Styled with Congo"
date = 2024-05-28T08:00:00+08:00
draft = false
description = "Explore the unique features of my blog powered by Hugo, the benefits of the Congo theme based on Tailwind CSS 3, and the versatility of Markdown functionalities, including Hugo's shortcodes."
summary = "Explore the unique features of my blog powered by Hugo, the benefits of the Congo theme based on Tailwind CSS 3, and the versatility of Markdown functionalities, including Hugo's shortcodes."
tags = ['Blogging', 'Hugo', 'Congo', 'Markdown', 'Shortcodes']
showWordCount = true
showEdit = true
editURL = "https://github.com/sylvanding/sylvanding.github.io/tree/main/content"
editAppendPath = true
+++

{{< alert "tag" >}}
**Hello and welcome to my brand-new blog! I'm thrilled to have you here and excited to share the news of its launch. After a lot of planning and designing, I've finally rolled out this platform as a space to share my thoughts, experiences, and knowledge.**
{{< /alert >}}

## Why I Chose Hugo and the Congo Theme

As a passionate blogger, the choice of platform and theme is crucial in shaping the reader experience and managing content efficiently. After evaluating several platforms and themes, I settled on Hugo with the Congo theme. Here’s why:

### The Power of Hugo: Speed and Flexibility Combined

> Hugo: The World’s Fastest Framework for Building Websites

[Hugo](https://gohugo.io/) is renowned for its incredible speed and efficiency. As a static site generator, it compiles pages almost instantaneously, making it perfect for bloggers who value quick build times. Furthermore, Hugo’s flexibility in handling various content types effortlessly has allowed me to structure my blog without the complexities often found in dynamic CMS platforms.

**Key Features of Hugo:**

- **Speed:** Hugo generates pages at lightning-fast speeds.
- **Customizability:** Extensive themes and tools for customization.
- **Security:** Being static, Hugo reduces common security risks associated with dynamic websites.

### Why the Congo Theme? Tailwind CSS at its Core

> Congo: A Powerful, Lightweight Theme for Hugo Built with Tailwind CSS

Choosing the [Congo](https://github.com/jpanther/congo) theme was a straightforward decision once I understood the advantages of Tailwind CSS. [Tailwind CSS version 3](https://tailwindcss.com/blog/tailwindcss-v3) offers an intuitive and powerful framework for designing unique and responsive layouts. Its utility-first approach means that almost any design is possible without leaving the comfort of your HTML.

**Benefits of the Congo Theme:**

- **Responsive Design:** Adjusts beautifully across all devices.
- **Customizable:** Easy to modify with utility classes.
- **Modern Aesthetics:** Sleek and clean design that keeps the focus on content.

## Leveraging Markdown for Enhanced Content Creation

Markdown is a lightweight markup language that makes it simpler to write formatted text but with plain text. By using Markdown, I ensure that my writing process remains straightforward yet powerful. Hugo further enriches this by integrating [Shortcodes](https://gohugo.io/content-management/shortcodes/)—a way to embed richer content elements directly within Markdown.

### Hugo Internal Shortcodes

#### Code Block Highlight

{{< highlight html "linenos=table,hl_lines=4 7-9" >}}
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Example HTML5 Document</title>
</head>
<body>
  <p>Test</p>
</body>
</html>
{{< /highlight >}}

### Congo Shortcodes

In addition to all the default Hugo shortcodes, Congo adds a few extras for additional functionality.

#### Alert

`alert` outputs its contents as a stylised message box within your article. It's useful for drawing attention to important information that you don't want the reader to miss.

The input is written in Markdown so you can format it however you please.

By default, the alert is presented with an exclaimation triangle icon. To change the icon, include the icon name in the shortcode. Check out the [icon shortcode](#icon) for more details on using icons.

**Example:**

```md
{{</* alert */>}}
**Warning!** This action is destructive!
{{</* /alert */>}}

{{</* alert "twitter" */>}}
Don't forget to [follow me](https://twitter.com/sylvanding) on Twitter.
{{</* /alert */>}}
```

{{< alert >}}
**Warning!** This action is destructive!
{{< /alert >}}
&nbsp;
{{< alert "twitter" >}}
Don't forget to [follow me](https://twitter.com/sylvanding) on Twitter.
{{< /alert >}}

#### Badge

`badge` outputs a styled badge component which is useful for displaying metadata.

**Example:**

```md
{{</* badge */>}}
New article!
{{</* /badge */>}}
```

{{< badge >}}
New article!
{{< /badge >}}

#### Button

`button` outputs a styled button component which can be used to highlight a primary action. It has three optional parameters:

<!-- prettier-ignore-start -->
|Parameter|Description|
|---|---|
|`href`|The URL that the button should link to.|
|`target`|The target of the link.|
|`download`|Whether browser should download the resource rather than navigate to the URL. The value of this parameter will be the name of the downloaded file.|
<!-- prettier-ignore-end -->

**Example:**

```md
{{</* button href="#button" target="_self" */>}}
Call to action
{{</* /button */>}}
```

{{< button href="#button" target="_self" >}}
Call to action
{{< /button >}}

#### Figure

Congo includes a `figure` shortcode for adding images to content. The shortcode replaces the base Hugo functionality in order to provide additional performance benefits.

When a provided image is a page resource, it will be optimised using Hugo Pipes and scaled in order to provide images appropriate to different device resolutions. If a static asset or URL to an external image is provided, it will be included as-is without any image processing by Hugo.

The `figure` shortcode accepts six parameters:

<!-- prettier-ignore-start -->
|Parameter|Description|
|---|---|
|`src`| **Required.** The local path/filename or URL of the image. When providing a path and filename, the theme will attempt to locate the image using the following lookup order: Firstly, as a [page resource](https://gohugo.io/content-management/page-resources/) bundled with the page; then an asset in the `assets/` directory; then finally, a static image in the `static/` directory.|
|`alt`|[Alternative text description](https://moz.com/learn/seo/alt-text) for the image.|
|`caption`|Markdown for the image caption, which will be displayed below the image.|
|`class`|Additional CSS classes to apply to the image.|
|`href`|URL that the image should be linked to.|
|`default`|Special parameter to revert to default Hugo `figure` behaviour. Simply provide `default=true` and then use normal [Hugo shortcode syntax](https://gohugo.io/content-management/shortcodes/#figure).|
<!-- prettier-ignore-end -->

Congo also supports automatic conversion of images included using standard Markdown syntax. Simply use the following format and the theme will handle the rest:

```md
![Alt text](image.jpg "Image caption")
```

**Example:**

```md
{{</* figure
    src="abstract.jpg"
    alt="Abstract purple artwork"
    caption="Photo by [Jr Korpa](https://unsplash.com/@jrkorpa) on [Unsplash](https://unsplash.com/)"
    */>}}

<!-- OR -->

![Abstract purple artwork](abstract.jpg "Photo by [Jr Korpa](https://unsplash.com/@jrkorpa) on [Unsplash](https://unsplash.com/)")
```

{{< figure src="abstract.jpg" alt="Abstract purple artwork" caption="Photo by [Jr Korpa](https://unsplash.com/@jrkorpa) on [Unsplash](https://unsplash.com/)" >}}

#### Chart

`chart` uses the Chart.js library to embed charts into articles using simple structured data. It supports a number of [different chart styles](https://www.chartjs.org/docs/latest/samples/) and everything can be configured from within the shortcode. Simply provide the chart parameters between the shortcode tags and Chart.js will do the rest.

Refer to the [official Chart.js docs](https://www.chartjs.org/docs/latest/general/) for details on syntax and supported chart types.

**Example:**

```js
{{</* chart */>}}
type: 'bar',
data: {
  labels: ['Tomato', 'Blueberry', 'Banana', 'Lime', 'Orange'],
  datasets: [{
    label: '# of votes',
    data: [12, 19, 3, 5, 3],
  }]
}
{{</* /chart */>}}
```

<!-- prettier-ignore-start -->
{{< chart >}}
type: 'bar',
data: {
  labels: ['Tomato', 'Blueberry', 'Banana', 'Lime', 'Orange'],
  datasets: [{
    label: '# of votes',
    data: [12, 19, 3, 5, 3],
  }]
}
{{< /chart >}}
<!-- prettier-ignore-end -->

#### Icon

`icon` outputs an SVG icon and takes the icon name as its only parameter. The icon is scaled to match the current text size.

**Example:**

```md
{{</* icon "github" */>}}
```

**Output:** {{< icon "github" >}}

Icons are populated using Hugo pipelines which makes them very flexible. Congo includes a number of built-in icons for social, links and other purposes.

Custom icons can be added by providing your own icon assets in the `assets/icons/` directory of your project. The icon can then be referenced in the shortcode by using the SVG filename without the `.svg` extension.

#### Katex

{{< katex >}}

The `katex` shortcode can be used to add mathematical expressions to article content using the KaTeX package. Refer to the online reference of [supported TeX functions](https://katex.org/docs/supported.html) for the available syntax.

To include mathematical expressions in an article, simply place the shortcode anywhere with the content. It only needs to be included once per article and KaTeX will automatically render any markup on that page. Both inline and block notation are supported.

Inline notation can be generated by wrapping the expression in `\\(` and `\\)` delimiters. Alternatively, block notation can be generated using `$$` delimiters.

Use the online reference of [supported TeX functions](https://katex.org/docs/supported.html) for the available syntax.

##### Inline notation

Inline notation can be generated by wrapping the expression in `\\(` and `\\)` delimiters.

**Example:**

```tex
% KaTeX inline notation
Inline notation: \\(\varphi = \dfrac{1+\sqrt5}{2}= 1.6180339887…\\)
```

Inline notation: \\(\varphi = \dfrac{1+\sqrt5}{2}= 1.6180339887…\\)

##### Block notation

Alternatively, block notation can be generated using `$$` delimiters. This will output the expression in its own HTML block.

**Example:**

```tex
% KaTeX block notation
$$
 \varphi = 1+\frac{1} {1+\frac{1} {1+\frac{1} {1+\cdots} } }
$$
```

$$
 \varphi = 1+\frac{1} {1+\frac{1} {1+\frac{1} {1+\cdots} } }
$$

#### Lead

`lead` is used to bring emphasis to the start of an article. It can be used to style an introduction, or to call out an important piece of information. Simply wrap any Markdown content in the `lead` shortcode.

**Example:**

```md
{{</* lead */>}}
When life gives you lemons, make lemonade.
{{</* /lead */>}}
```

{{< lead >}}
When life gives you lemons, make lemonade.
{{< /lead >}}

#### Mermaid

`mermaid` allows you to draw detailed diagrams and visualisations using text. It uses Mermaid under the hood and supports a wide variety of diagrams, charts and other output formats.

Simply write your Mermaid syntax within the `mermaid` shortcode and let the plugin do the rest.

Refer to the [official Mermaid docs](https://mermaid-js.github.io/) for details on syntax and supported diagram types.

**Example:**

```md
{{</* mermaid */>}}
graph LR;
A[Lemons]-->B[Lemonade];
B-->C[Profit]
{{</* /mermaid */>}}
```

{{< mermaid >}}
graph LR;
A[Lemons]-->B[Lemonade];
B-->C[Profit]
{{< /mermaid >}}

### Custom Shortcodes

To create a shortcode, place an HTML template in the `layouts/shortcodes` directory of your [source organization](https://gohugo.io/getting-started/directory-structure/). Consider the file name carefully since the shortcode name will mirror that of the file but without the `.html` extension. For example, `layouts/shortcodes/myshortcode.html` will be called with `{{</* myshortcode */>}}`.

#### Music Block

Music Block shortcodes require [APlayer](https://github.com/MoePlayer/APlayer) and [MetingJS](https://github.com/metowolf/MetingJS). Place the downloaded js and css files in the `assets/plugins` directory:

```html
<!-- require APlayer -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/aplayer/dist/APlayer.min.css">
<script src="https://cdn.jsdelivr.net/npm/aplayer/dist/APlayer.min.js"></script>
<!-- require MetingJS -->
<script src="https://cdn.jsdelivr.net/npm/meting@2/dist/Meting.min.js"></script>
```

Then create a `layouts/shortcodes/meting.html` file:

```md
<meting-js server="{{ .Get "server" }}" type="{{ .Get "type" }}" id="{{ .Get "id" }}"></meting-js>
{{/* MetingJS@2.0.x */}}
{{ if .Site.Params.MetingJS | default false }}
  <!-- require APlayer -->
  {{ with resources.Get "plugins/APlayer/APlayer.min.css" }}
    <link rel="stylesheet" href="{{ .RelPermalink }}" />
  {{ end }}
  {{ with resources.Get "plugins/APlayer/APlayer.min.js" }}
    <script src="{{ .RelPermalink }}"></script>
  {{ end }}
  <!-- require MetingJS -->
  {{ with resources.Get "plugins/Meting/Meting.min.js" }}
    <script src="{{ .RelPermalink }}"></script>
  {{ end }}
{{ end }}
```

**Example:**

```md
{{</* meting server="netease" type="song" id="2124385868" */>}}
```

{{< meting server="netease" type="song" id="5179543" >}}

## Conclusion

**Thank you for visiting my new blog! I'm excited about the journey ahead and look forward to engaging with a community of tech enthusiasts and developers. Stay tuned for more posts where I'll dive deeper into various technical topics, share development tips, and much more. Don’t forget to subscribe and join the conversation in the comments section below!**