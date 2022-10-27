#![cfg_attr(debug_assertions, allow(dead_code, unused_variables,))]
use std::{
    cell::Cell,
    collections::{HashMap, HashSet, VecDeque},
    io::Write,
    time::{Duration, Instant},
};

use anyhow::Context;
use egui::{
    vec2, CentralPanel, Color32, FontData, FontDefinitions, FontFamily, Frame, Key, Label,
    RichText, ScrollArea, Slider, TextStyle, Vec2,
};
use egui_extras::RetainedImage;
use image::ImageFormat;
use tokio_stream::StreamExt;

const MESSAGE_LIMIT: usize = 5;

enum View {
    Main,
    Settings,
}

struct OnScreenKappas {
    decorations: bool,
    config: Configuration,
    client: chat::Client,
    messages: MessageCache,
    badges: BadgeMap,
    images: ImageCache,
    mention: Mention,
    history: History,
    view: View,
}

impl OnScreenKappas {
    fn new(
        client: chat::Client,
        images: ImageCache,
        badges: BadgeMap,
        config: Configuration,
        channel: String,
        history: History,
    ) -> Self {
        Self {
            decorations: false,
            config,
            client,
            messages: MessageCache::new(),
            badges,
            images,
            mention: Mention::new(channel),
            history,
            view: View::Main,
        }
    }

    fn handle_keypresses(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) -> bool {
        if ctx.input().key_pressed(Key::F12) {
            let hover = ctx.debug_on_hover();
            ctx.set_debug_on_hover(!hover);
        }

        if ctx.input().key_pressed(Key::Q) {
            frame.close();
            return false;
        }

        if ctx.input().key_pressed(Key::F1) {
            self.decorations = !self.decorations;
            frame.set_decorations(self.decorations);
            frame.set_hittest(self.decorations)
        }

        if ctx.input().key_pressed(Key::T) {
            self.config.timestamps = !self.config.timestamps
        }

        if ctx.input().key_pressed(Key::B) {
            self.config.badges = !self.config.badges
        }

        if ctx.input().key_pressed(Key::H) {
            self.config.highlights = !self.config.highlights
        }

        if ctx.input().key_pressed(Key::F2) {
            self.view = if matches!(self.view, View::Main) {
                View::Settings
            } else {
                View::Main
            };
        }

        enum Either<L, R> {
            Left(L),
            Right(R),
        }
        enum Fade {
            In,
            Out,
        }

        enum Zoom {
            In,
            Out,
        }

        let mouse_wheel = ctx.input().events.iter().find_map(|ev| match ev {
            &egui::Event::Zoom(delta) if delta != 1.0 => (delta > 1.0)
                .then_some(Zoom::In)
                .or(Some(Zoom::Out))
                .map(Either::Left),
            &egui::Event::Scroll(delta) => {
                let delta = (delta.y / 200.0).exp();
                if delta == 1.0 {
                    return None;
                }
                (delta > 1.0)
                    .then_some(Fade::In)
                    .or(Some(Fade::Out))
                    .map(Either::Right)
            }
            _ => None,
        });

        let old = self.config.scale;

        match mouse_wheel {
            Some(Either::Left(Zoom::In)) => self.config.scale = (self.config.scale + 0.1).min(3.0),
            Some(Either::Left(Zoom::Out)) => self.config.scale = (self.config.scale - 0.1).max(1.0),

            Some(Either::Right(Fade::In)) => self.config.alpha = (self.config.alpha + 0.1).min(1.0),
            Some(Either::Right(Fade::Out)) => {
                self.config.alpha = (self.config.alpha - 0.1).max(0.0)
            }

            None => {}
        }

        if self.config.scale != old {
            ctx.set_pixels_per_point(self.config.scale);
            ctx.request_repaint();
        }

        true
    }

    fn update_logic(&mut self, _ctx: &egui::Context) {
        self.client.poll(
            &mut self.messages.messages,
            |msg| self.history.append(msg),
            |msg| MessageCache::prepare(msg, &self.mention, &self.badges),
        );

        for msg in self.client.clears.try_iter() {
            match msg {
                chat::Clear::All => {
                    self.messages.clear_all();
                }
                chat::Clear::Chat(user) => self.messages.clear_user(user),
                chat::Clear::Message { id } => self.messages.clear_msg(id),
            }
        }

        self.images.poll();
    }

    fn render(&mut self, ctx: &egui::Context) {
        match self.view {
            View::Main => self.render_main(ctx),
            View::Settings => self.render_settings(ctx),
        }
    }

    fn render_main(&mut self, ctx: &egui::Context) {
        CentralPanel::default()
            .frame(Frame::none().fill(egui::Rgba::from_black_alpha(self.config.alpha).into()))
            .show(ctx, |ui| {
                ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .always_show_scroll(false)
                    .scroll2([false, false])
                    .show(ui, |ui| {
                        let now = time::OffsetDateTime::now_local().expect("system clock");

                        if !self.history.dirty {
                            for msg in &self.history.messages {
                                Self::show_message(msg, &mut self.images, self.config, ui);
                            }
                        }

                        for msg in self.messages.iter_mut() {
                            match &mut msg.view_state {
                                state @ MessageViewState::Show => {
                                    if now - msg.timestamp >= Duration::from_secs(20) {
                                        *state =
                                            MessageViewState::Fade(ctx.animate_bool_with_time(
                                                egui::Id::new(msg.inner.id),
                                                true,
                                                3.0,
                                            ));
                                    }
                                }
                                MessageViewState::Fade(op) if *op > 0.0 => {
                                    *op = ctx.animate_bool_with_time(
                                        egui::Id::new(msg.inner.id),
                                        false,
                                        3.0,
                                    );
                                }
                                s @ MessageViewState::Fade(..) => {
                                    *s = MessageViewState::Hide;
                                    continue;
                                }
                                MessageViewState::Hide => continue,
                                MessageViewState::Removed => {}
                            }

                            Self::show_message(msg, &mut self.images, self.config, ui);
                        }
                    });
            });
    }

    fn render_settings(&mut self, ctx: &egui::Context) {
        CentralPanel::default()
            .frame(Frame::none().fill(egui::Rgba::from_black_alpha(self.config.alpha).into()))
            .show(ctx, |ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.monospace("Transparency");
                        ui.add(Slider::new(&mut self.config.alpha, 0.0..=1.0).step_by(0.05));
                    });

                    let old = self.config.scale;

                    ui.horizontal(|ui| {
                        ui.monospace("Zoom level");

                        ui.horizontal(|ui| {
                            if ui.small_button("-").clicked() {
                                self.config.scale = (self.config.scale - 0.1).max(1.0);
                            }
                            ui.label(format!("{:0.2?}", self.config.scale));
                            if ui.small_button("+").clicked() {
                                self.config.scale = (self.config.scale + 0.1).min(3.0);
                            }
                        });
                    });

                    if old != self.config.scale {
                        ctx.set_pixels_per_point(self.config.scale);
                        ctx.request_repaint();
                    }

                    ui.horizontal(|ui| {
                        ui.monospace("cull duration");

                        ui.add(
                            Slider::new(&mut self.config.cull_duration, 1..=5 * 60).step_by(10.0),
                        )
                    })
                });
            });
    }

    fn show_message(
        msg: &PreparedMessage,
        images: &mut ImageCache,
        config: Configuration,
        ui: &mut egui::Ui,
    ) {
        static TIME_FORMAT: &[time::format_description::FormatItem<'static>] =
            time::macros::format_description!("[hour]:[minute]:[second]");

        let resp = ui
            .horizontal_wrapped(|ui| {
                ui.scope(|ui| {
                    let width = ui
                        .fonts()
                        .glyph_width(&TextStyle::Body.resolve(ui.style()), ' ');
                    ui.spacing_mut().item_spacing.x = width;

                    if config.timestamps {
                        let ts = msg
                            .timestamp
                            .format(&TIME_FORMAT)
                            .expect("formattable time");

                        ui.small(ts);
                    }

                    if config.badges {
                        if let Some(badge) = msg.badges.first() {
                            if let Some(img) = images.get(badge) {
                                img.show_size(ui, vec2(8.0, 8.0));
                            }
                        }
                    }

                    ui.add(Label::new({
                        let mut rt = RichText::new(&msg.inner.sender.name).color(msg.color);
                        if msg.is_mention && config.highlights {
                            // TODO create an alpha mask for the line instead of just changing this
                            let background_color = Color32::from_rgba_premultiplied(64, 0, 0, 224);
                            rt = rt.background_color(background_color)
                        }
                        rt
                    }));

                    if matches!(msg.view_state, MessageViewState::Removed) {
                        ui.colored_label(Color32::RED, "*removed*");
                        return;
                    }

                    for span in &msg.message_spans {
                        match span {
                            TextSpan::Text(s) => {
                                ui.add(Label::new(RichText::new(s).strong()));
                            }
                            TextSpan::Link(..) => {
                                ui.add(Label::new(
                                    RichText::new("[link]").color(Color32::LIGHT_BLUE),
                                ));
                            }
                            TextSpan::Emote(s) => {
                                for url in s.as_urls() {
                                    if let Some(img) = images.get(url) {
                                        img.show_size(ui, vec2(16.0, 16.0));
                                        break;
                                    }
                                }
                            }
                        }
                    }
                });
            })
            .response;

        ui.put(resp.rect, move |ui: &mut egui::Ui| {
            Frame::none()
                .fill(
                    egui::Rgba::from_black_alpha(
                        config.alpha
                            - egui::lerp(0.0..=msg.opacity(), config.alpha).min(config.alpha),
                    )
                    .into(),
                )
                .show(ui, |ui| ui.add_sized(ui.available_size(), Label::new("")))
                .response
        });
    }
}

#[derive(Default, Copy, Clone, serde::Serialize, serde::Deserialize)]
enum MessageViewState {
    #[default]
    Show,
    Fade(f32),
    Hide,
    Removed,
}

struct MessageCache {
    messages: Ring<PreparedMessage>,
}

impl MessageCache {
    fn new() -> Self {
        Self {
            messages: Ring::new(MESSAGE_LIMIT),
        }
    }

    pub fn clear_all(&mut self) {
        for msg in &mut self.messages.buf {
            msg.view_state = MessageViewState::Removed
        }
    }

    pub fn clear_user(&mut self, user: chat::User) {
        for msg in self
            .messages
            .buf
            .iter_mut()
            .filter(|msg| msg.inner.sender.id == user.id)
        {
            msg.view_state = MessageViewState::Removed
        }
    }

    pub fn clear_msg(&mut self, id: uuid::Uuid) {
        if let Some(msg) = self.messages.buf.iter_mut().find(|msg| msg.inner.id == id) {
            msg.view_state = MessageViewState::Removed
        }
    }

    fn prepare(msg: chat::Message, mention: &Mention, badges: &BadgeMap) -> PreparedMessage {
        PreparedMessage {
            timestamp: time::OffsetDateTime::now_local().expect("system should have a clock"),
            color: Self::parse_color(&msg.tags),
            badges: Self::parse_badges(&msg.tags, badges),
            message_spans: Self::parse_emotes(&msg.tags, &msg.data),
            is_mention: Self::check_for_mentions(&msg, mention),
            inner: msg,
            view_state: MessageViewState::default(),
        }
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut PreparedMessage> + ExactSizeIterator {
        self.messages.iter_mut()
    }

    fn check_for_mentions(msg: &chat::Message, channel: &Mention) -> bool {
        msg.data.split_whitespace().any(|c| channel == c)
    }

    fn parse_emotes(tags: &chat::Tags, data: &str) -> Vec<TextSpan> {
        let chars = data.trim_end().chars().collect::<Vec<_>>();
        let mut emotes = tags
            .get("emotes")
            .into_iter()
            .flat_map(|s| s.split('/'))
            .flat_map(|s| s.split_once(':'))
            .flat_map(|(emote, range)| {
                range
                    .split(',')
                    .flat_map(|c| c.split_once('-'))
                    .flat_map(|(start, end)| Some((start.parse().ok()?, end.parse().ok()?)))
                    .zip(std::iter::repeat(emote))
                    .map(|((start, end), kind): ((usize, usize), _)| {
                        (kind, (start, end - start + 1))
                    })
            })
            .collect::<Vec<(&str, (usize, usize))>>();

        emotes.sort_unstable_by_key(|(_, r)| *r);

        let trim = |data: &[char]| {
            let tail = data
                .iter()
                .rev()
                .take_while(|c| c.is_ascii_whitespace())
                .count();
            data.iter()
                .take(data.len() - tail)
                .skip_while(|c| c.is_ascii_whitespace())
                .collect()
        };

        let mut spans = vec![];
        let mut cursor = 0;

        for (emote, (start, end)) in emotes {
            if start != cursor {
                let text = trim(&chars[cursor..start]);
                spans.push(TextSpan::Text(text));
            }

            spans.push(TextSpan::Emote(EmoteSpan::new(emote)));
            cursor = start + end;
        }

        if cursor != chars.len() {
            let text = trim(&chars[cursor..]);
            spans.push(TextSpan::Text(text));
        }

        let len = spans.len();
        spans
            .into_iter()
            .fold(Vec::with_capacity(len), |mut out, span| {
                match span {
                    TextSpan::Text(text) => out.extend(text.split_whitespace().map(|s| {
                        let ctor = url::Url::parse(s)
                            .ok()
                            .filter(|c| matches!(c.scheme(), "http" | "https"))
                            .map(|_| TextSpan::Link as fn(String) -> TextSpan)
                            .unwrap_or_else(|| TextSpan::Text);
                        ctor(s.to_string())
                    })),
                    span @ TextSpan::Emote(_) => out.push(span),
                    _ => {}
                };
                out
            })
    }

    fn parse_badges(tags: &chat::Tags, badges: &BadgeMap) -> Vec<String> {
        tags.get("badges")
            .into_iter()
            .flat_map(|badges| badges.split(','))
            .flat_map(|badge| badge.split_once('/'))
            .flat_map(|(badge, version)| badges.get(badge, version))
            .map(ToString::to_string)
            .collect()
    }

    fn parse_color(tags: &chat::Tags) -> egui::Color32 {
        struct Color(u8, u8, u8);
        impl std::str::FromStr for Color {
            type Err = &'static str;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let s = match s.len() {
                    7 => &s[1..],
                    6 => s,
                    _ => return Err("invalid color"),
                };

                let color = u32::from_str_radix(s, 16).map_err(|_| "invalid hex digit")?;
                let (r, g, b) = (
                    ((color >> 16) & 0xFF) as _,
                    ((color >> 8) & 0xFF) as _,
                    (color & 0xFF) as _,
                );
                Ok(Self(r, g, b))
            }
        }
        impl Default for Color {
            fn default() -> Self {
                Self(0xFF, 0xFF, 0xFF)
            }
        }

        let Color(r, g, b) = tags
            .get_parsed("color")
            .transpose()
            .ok()
            .flatten()
            .unwrap_or_default();

        egui::Color32::from_rgb(r, g, b)
    }
}

struct Mention {
    data: String,
}

impl Mention {
    fn new(data: String) -> Self {
        Self {
            data: if data.starts_with('@') {
                data
            } else {
                format!("@{data}")
            },
        }
    }
}

impl<T> PartialEq<T> for Mention
where
    T: AsRef<str> + ?Sized,
{
    fn eq(&self, other: &T) -> bool {
        let other = other.as_ref();
        self.data.eq_ignore_ascii_case(other) || self.data[1..].eq_ignore_ascii_case(other)
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct PreparedMessage {
    timestamp: time::OffsetDateTime,
    color: egui::Color32,
    inner: chat::Message,
    badges: Vec<String>,
    message_spans: Vec<TextSpan>,
    is_mention: bool,
    view_state: MessageViewState,
}

impl PreparedMessage {
    fn opacity(&self) -> f32 {
        match self.view_state {
            MessageViewState::Fade(op) => op,
            _ => 1.0,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum TextSpan {
    Text(String),
    Link(String),
    Emote(EmoteSpan),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct EmoteSpan {
    urls: [String; 2],
}

impl EmoteSpan {
    fn new(id: impl ToString) -> Self {
        let id = id.to_string();

        Self {
            urls: [
                format!(
                    "https://static-cdn.jtvnw.net/emoticons/v2/{id}/{format}/{theme_mode}/{scale}",
                    id = id,
                    format = "animated",
                    theme_mode = "dark",
                    scale = "1.0",
                ),
                format!(
                    "https://static-cdn.jtvnw.net/emoticons/v2/{id}/{format}/{theme_mode}/{scale}",
                    id = id,
                    format = "static",
                    theme_mode = "dark",
                    scale = "1.0",
                ),
            ],
        }
    }

    fn as_urls(&self) -> impl Iterator<Item = &str> {
        self.urls.iter().map(|c| &**c)
    }
}

#[derive(Debug)]
struct BadgeMap {
    map: HashMap<String, HashMap<String, String>>,
}

impl BadgeMap {
    fn load_from_static_json() -> anyhow::Result<Self> {
        #[derive(serde::Deserialize)]
        struct Root {
            data: Vec<Data>,
        }
        #[derive(serde::Deserialize)]
        struct Data {
            set_id: String,
            versions: Vec<Version>,
        }

        #[derive(serde::Deserialize)]
        struct Version {
            id: String,
            image_url_1x: String,
        }

        const GLOBAL_BADGES_JSON: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/",
            "global_badges.json"
        ));

        let data: Root = serde_json::from_str(GLOBAL_BADGES_JSON)?;

        Ok(Self {
            map: data.data.into_iter().fold(
                HashMap::<_, HashMap<_, _>>::new(),
                |mut map, Data { set_id, versions }| {
                    let sub = map.entry(set_id).or_default();
                    for Version { id, image_url_1x } in versions {
                        sub.insert(id, image_url_1x);
                    }
                    map
                },
            ),
        })
    }

    fn get(&self, badge: &str, version: &str) -> Option<&str> {
        self.map.get(badge)?.get(version).map(|v| &**v)
    }
}

struct ImageCache {
    map: HashMap<String, Image>,
    fetcher: ImageFetcher,
}

impl ImageCache {
    fn new(fetcher: ImageFetcher) -> Self {
        Self {
            map: HashMap::default(),
            fetcher,
        }
    }

    fn get(&mut self, url: &str) -> Option<&Image> {
        match self.map.get(url) {
            Some(img) => Some(img),
            None => {
                self.fetcher.request(url);
                None
            }
        }
    }

    fn poll(&mut self) {
        for (k, v) in self.fetcher.poll() {
            self.map.insert(k, v);
        }
    }
}

#[derive(Clone)]
struct ImageFetcher {
    submit: flume::Sender<String>,
    produce: flume::Receiver<(String, Image)>,
}

impl ImageFetcher {
    fn spawn(repaint: impl Repaint + 'static) -> Self {
        let (submit, submit_rx) = flume::unbounded::<String>();
        let (produce_tx, produce) = flume::unbounded();

        crate::runtime::spawn(async move {
            let mut seen = HashSet::<String>::new();
            let mut stream = submit_rx.into_stream();
            let client = reqwest::Client::new();

            while let Some(url) = stream.next().await {
                if !seen.insert(url.clone()) {
                    continue;
                }

                let client = client.clone();
                let tx = produce_tx.clone();
                let repaint = repaint.clone();
                tokio::spawn(async move {
                    if let Some(data) = Self::fetch(client, &url).await {
                        tokio::task::spawn_blocking(move || match Self::load_image(&url, data) {
                            Ok(img) => {
                                repaint.repaint();
                                let _ = tx.send((url, img));
                            }
                            Err(err) => {
                                eprintln!("error: {err}")
                            }
                        });
                    }
                });
            }
        });

        Self { submit, produce }
    }

    fn request(&self, url: &str) {
        let _ = self.submit.send(url.to_string());
    }

    fn poll(&mut self) -> impl Iterator<Item = (String, Image)> + '_ {
        self.produce.try_iter()
    }

    async fn fetch(client: reqwest::Client, url: &str) -> Option<Vec<u8>> {
        let resp = client.get(url).send().await.ok()?;

        if resp.status().as_u16() == 404 {
            return None;
        }

        resp.bytes().await.ok().map(|v| v.to_vec())
    }

    fn load_image(url: &str, data: Vec<u8>) -> anyhow::Result<Image> {
        let img = match image::guess_format(&data[..data.len().min(128)])
            .map_err(|err| anyhow::anyhow!("cannot guess format for '{url}': {err}"))?
        {
            image::ImageFormat::Png => {
                let dec = image::codecs::png::PngDecoder::new(&*data).map_err(|err| {
                    anyhow::anyhow!("expected png, got something else for '{url}': {err}")
                })?;

                if dec.is_apng() {
                    AnimatedImage::load_apng(url, &data).map(Image::Animated)?
                } else {
                    Self::load_retained_image(url, &data).map(Image::Static)?
                }
            }
            image::ImageFormat::Jpeg => Self::load_retained_image(url, &data).map(Image::Static)?,
            image::ImageFormat::Gif => AnimatedImage::load_gif(url, &data).map(Image::Animated)?,
            fmt => anyhow::bail!("unsupported format for '{url}': {fmt:?}"),
        };

        Ok(img)
    }

    fn load_retained_image(url: &str, data: &[u8]) -> anyhow::Result<egui_extras::RetainedImage> {
        RetainedImage::from_image_bytes(url, data)
            .map_err(|err| anyhow::anyhow!("cannot load '{url}': {err}"))
    }
}

enum Image {
    Static(egui_extras::RetainedImage),
    Animated(AnimatedImage),
}

impl Image {
    fn show_size(&self, ui: &mut egui::Ui, size: Vec2) {
        match self {
            Image::Static(img) => {
                img.show_size(ui, size);
            }
            Image::Animated(img) => {
                let dt = ui.ctx().input().stable_dt.min(0.1);
                if let Some((img, delay)) = img.frame(dt) {
                    img.show_size(ui, size);
                    ui.ctx().request_repaint_after(delay);
                }
            }
        }
    }
}

struct AnimatedImage {
    frames: Vec<egui_extras::RetainedImage>,
    intervals: Vec<Duration>,
    position: Cell<usize>,
    last: Cell<Option<Instant>>,
}

impl AnimatedImage {
    fn frame(&self, dt: f32) -> Option<(&egui_extras::RetainedImage, Duration)> {
        let pos = self.position.get();
        let delay = self.intervals.get(pos)?;

        match self.last.get() {
            Some(last) if last.elapsed().as_secs_f32() >= delay.as_secs_f32() - dt => {
                self.position.set((pos + 1) % self.frames.len());
                self.last.set(Some(Instant::now()));
            }
            Some(..) => {}
            None => {
                self.last.set(Some(Instant::now()));
            }
        }

        self.frames.get(pos).map(|frame| (frame, *delay))
    }

    fn load_apng(url: &str, data: &[u8]) -> anyhow::Result<Self> {
        use image::ImageDecoder as _;
        let dec = image::codecs::png::PngDecoder::new(data)?;
        anyhow::ensure!(dec.is_apng(), "expected an animated png");
        Self::load_frames(url, dec.total_bytes() as _, dec.apng())
    }

    fn load_gif(url: &str, data: &[u8]) -> anyhow::Result<Self> {
        use image::ImageDecoder as _;
        let dec = image::codecs::gif::GifDecoder::new(data)?;
        Self::load_frames(url, dec.total_bytes() as _, dec)
    }

    fn load_frames<'a>(
        name: &str,
        size_hint: usize,
        decoder: impl image::AnimationDecoder<'a>,
    ) -> anyhow::Result<Self> {
        let mut buf = std::io::Cursor::new(Vec::with_capacity(size_hint));
        let (mut frames, mut intervals) = (vec![], vec![]);

        for (i, frame) in decoder.into_frames().enumerate() {
            let frame = frame?;
            let delay = Duration::from(frame.delay());
            frame.buffer().write_to(&mut buf, ImageFormat::Png)?;
            buf.flush().expect("flush image transcode");
            let pos = buf.position();
            buf.set_position(0);
            let image = ImageFetcher::load_retained_image(
                &format!("{name}_{i}"),
                &buf.get_ref()[..pos as usize],
            )
            .with_context(|| anyhow::anyhow!("cannot decode frame {i}"))?;

            frames.push(image);
            intervals.push(delay);
        }

        Ok(Self {
            frames,
            intervals,
            position: Cell::new(0),
            last: Cell::new(None),
        })
    }
}

impl eframe::App for OnScreenKappas {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        if !self.handle_keypresses(ctx, frame) {
            return;
        }
        self.update_logic(ctx);
        self.render(ctx);
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        storage.set_string(
            CONFIGURATION_KEY,
            serde_json::to_string(&self.config).expect("valid json"),
        );

        storage.set_string(HISTORY_KEY, self.history.to_json());
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> egui::Rgba {
        egui::Rgba::from_black_alpha(self.config.alpha)
    }
}

pub trait Repaint: Clone + Send + Sync {
    fn repaint(&self) {}
}

impl Repaint for () {}

impl Repaint for egui::Context {
    fn repaint(&self) {
        self.request_repaint()
    }
}

pub struct Ring<T> {
    buf: VecDeque<T>,
    max: usize,
}

impl<T> Ring<T> {
    fn new(max: usize) -> Self {
        assert!(max > 0);
        Self {
            buf: VecDeque::new(),
            max,
        }
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> + ExactSizeIterator {
        self.buf.iter_mut()
    }

    fn push(&mut self, item: T) {
        while self.len() == self.cap() {
            self.buf.pop_front();
        }
        self.buf.push_back(item);
    }

    fn len(&self) -> usize {
        self.buf.len()
    }

    fn cap(&self) -> usize {
        self.max
    }
}

impl<T> Extend<T> for Ring<T> {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        iter.into_iter().for_each(|item| self.push(item));
    }
}

mod runtime {
    use std::future::Future;

    static TOKIO_HANDLE: once_cell::sync::OnceCell<tokio::runtime::Handle> =
        once_cell::sync::OnceCell::new();

    pub fn start() -> std::io::Result<()> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;

        let handle = rt.handle().clone();
        let _thread = std::thread::spawn(move || {
            rt.block_on(std::future::pending::<()>());
        });

        TOKIO_HANDLE.get_or_init(|| handle);

        Ok(())
    }

    pub fn enter_context(f: impl FnOnce()) {
        let _g = TOKIO_HANDLE.get().expect("initialization").enter();
        f();
    }

    pub fn spawn<T>(fut: impl Future<Output = T> + Send + Sync + 'static) -> flume::Receiver<T>
    where
        T: Send + Sync + 'static,
    {
        let (tx, rx) = flume::bounded(1); // not-sync
        enter_context(|| {
            tokio::task::spawn(async move {
                let res = fut.await;
                let _ = tx.send(res);
            });
        });
        rx
    }
}

mod chat {
    use std::{borrow::Cow, collections::BTreeMap};

    use tokio::io::{AsyncBufReadExt, AsyncWriteExt as _, BufReader};

    use crate::{Repaint, Ring};

    #[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct Tags {
        pub inner: BTreeMap<String, String>,
    }

    impl Tags {
        fn parse(input: &str) -> Option<Self> {
            if !input.starts_with('@') {
                return None;
            }

            let inner = input[1..]
                .split_terminator(';')
                .flat_map(|tag| tag.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();

            Some(Self { inner })
        }

        pub fn get(&self, key: &str) -> Option<&str> {
            self.inner.get(key).map(|v| &**v)
        }

        pub fn get_parsed<T>(&self, key: &str) -> Option<Result<T, T::Err>>
        where
            T: std::str::FromStr,
            T::Err: std::fmt::Debug,
        {
            self.get(key).map(<str>::parse)
        }
    }

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct User {
        pub name: String,
        pub id: i64,
    }

    #[derive(Debug, Clone)]
    pub enum Clear {
        All,
        Chat(User),
        Message { id: uuid::Uuid },
    }

    impl Clear {
        fn parse(input: &str) -> Option<Self> {
            Self::parse_clear_message(input).or_else(|| Self::parse_clear_chat(input))
        }

        fn parse_clear_chat(input: &str) -> Option<Self> {
            // @room-id=12345678;tmi-sent-ts=1642715695392 :tmi.twitch.tv CLEARCHAT #dallas
            // @ban-duration=<duration>;room-id=<room-id>;target-user-id=<user-id>;tmi-sent-ts=<timestamp> :tmi.twitch.tv CLEARCHAT #dallas :ronni

            use once_cell::sync::Lazy;
            use regex::{Match, Regex};
            static MESSAGE_PATTERN: Lazy<Regex> = Lazy::new(|| {
                Regex::new(
                        r##"^(?P<tags>@.*?)\s(?::tmi.twitch.tv)\sCLEARCHAT\s(?:#.*?)(?:\s:(?P<data>.*?))?(?:\r\n)?$"##,
                    ).expect("valid regex")
            });

            let captures = MESSAGE_PATTERN.captures(input)?;
            macro_rules! extract {
                ($key:expr) => {
                    captures.name($key).as_ref().map(Match::as_str)
                };
            }

            let tags = extract!("tags").and_then(Tags::parse)?;

            match extract!("data") {
                Some(name) => Some(Self::Chat(User {
                    name: name.to_string(),
                    id: tags.get_parsed("target-user-id")?.ok()?,
                })),
                None => Some(Self::All),
            }
        }

        fn parse_clear_message(input: &str) -> Option<Self> {
            // @login=<login>;room-id=<room-id>;target-msg-id=<target-msg-id>;tmi-sent-ts=<timestamp> :tmi.twitch.tv CLEARMSG #dallas :HeyGuys

            use once_cell::sync::Lazy;
            use regex::{Match, Regex};
            static MESSAGE_PATTERN: Lazy<Regex> = Lazy::new(|| {
                Regex::new(
                        r##"^(?P<tags>@.*?)\s(?::tmi.twitch.tv)\sCLEARMSG\s(?:#.*?)\s(:?:.*?)(:?\r\n)$"##,
                    )
                    .expect("valid regex")
            });

            let captures = MESSAGE_PATTERN.captures(input)?;
            macro_rules! extract {
                ($key:expr) => {
                    captures.name($key).as_ref().map(Match::as_str)
                };
            }

            extract!("tags")
                .and_then(Tags::parse)?
                .get_parsed("target-msg-id")?
                .ok()
                .map(|id| Self::Message { id })
        }
    }

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct Message {
        pub id: uuid::Uuid,
        pub tags: Tags,
        pub sender: User,
        pub data: String,
    }

    impl Message {
        fn parse(input: &str) -> Option<Self> {
            use once_cell::sync::Lazy;
            use regex::{Match, Regex};
            static MESSAGE_PATTERN: Lazy<Regex> = Lazy::new(|| {
                Regex::new(
                        r##"^(?P<tags>@.*?)\s(?::(?P<sender>.*?)!.*?)\sPRIVMSG\s(?:#.*?)\s:(?P<data>.*?)(?:\r\n)?$"##,
                    ).expect("valid regex")
            });

            let captures = MESSAGE_PATTERN.captures(input)?;
            macro_rules! extract {
                ($key:expr) => {
                    captures.name($key).as_ref().map(Match::as_str)
                };
            }

            let tags = extract!("tags").and_then(Tags::parse)?;

            Some(Self {
                id: tags.get_parsed("id")?.ok()?,
                sender: User {
                    name: extract!("display-name")
                        .or_else(|| extract!("sender"))
                        .map(ToString::to_string)?,
                    id: tags.get_parsed("room-id").transpose().ok().flatten()?,
                },
                data: extract!("data").map(ToString::to_string)?,
                tags,
            })
        }
    }

    #[derive(Clone)]
    pub struct Client {
        pub messages: flume::Receiver<Message>,
        pub clears: flume::Receiver<Clear>,
    }

    impl Client {
        pub fn poll<T>(
            &self,
            messages: &mut Ring<T>,
            mut history: impl FnMut(T),
            map: impl Fn(Message) -> T,
        ) where
            T: Clone,
        {
            for msg in self.messages.try_iter() {
                let msg = map(msg);
                messages.push(msg.clone());
                history(msg)
            }
        }
    }

    pub fn start(repaint: impl Repaint + 'static, channel: String) -> Client {
        let (messages_tx, messages) = flume::unbounded();
        let (clears_tx, clears) = flume::unbounded();
        let client = Client { messages, clears };

        crate::runtime::spawn::<anyhow::Result<()>>(async move {
            let mut stream = tokio::net::TcpStream::connect("irc.chat.twitch.tv:6667").await?;

            for registration in [
                "CAP REQ :twitch.tv/membership\r\n",
                "CAP REQ :twitch.tv/tags\r\n",
                "CAP REQ :twitch.tv/commands\r\n",
                // TODO randomly generate this, we're going to be read-only
                "PASS justinfan1234\r\n",
                "NICK justinfan1234\r\n",
            ] {
                stream.write_all(registration.as_bytes()).await?;
            }

            stream.flush().await?;

            let (read, mut write) = stream.into_split();
            let mut lines = BufReader::new(read).lines();

            while let Ok(Some(line)) = lines.next_line().await {
                eprintln!("{}", line.escape_debug());

                let channel = if channel.starts_with('#') {
                    Cow::Borrowed(&channel)
                } else {
                    Cow::Owned(format!("#{channel}"))
                };

                if line == "PING :tmi.twitch.tv" {
                    write.write_all(b"PONG tmi.twitch.tv\r\n").await?;
                    write.flush().await?;
                }

                if line == ":tmi.twitch.tv 376 justinfan1234 :>" {
                    write
                        .write_all(format!("JOIN {channel}\r\n").as_bytes())
                        .await?;
                    write.flush().await?;
                }

                if let Some(msg) = Message::parse(&line) {
                    repaint.repaint();
                    if messages_tx.send(msg).is_err() {
                        break;
                    }
                }

                if let Some(clear) = Clear::parse(&line) {
                    repaint.repaint();
                    if clears_tx.send(clear).is_err() {
                        break;
                    }
                }
            }

            Ok(())
        });

        client
    }
}

#[derive(Default, serde::Serialize, serde::Deserialize)]
struct History {
    messages: VecDeque<PreparedMessage>,
    #[serde(default, skip)]
    dirty: bool,
}

impl History {
    fn append(&mut self, msg: PreparedMessage) {
        self.dirty = true;

        if self.messages.len() == MESSAGE_LIMIT {
            self.messages.pop_front();
        }
        self.messages.push_back(msg)
    }

    fn from_json(json: &str) -> Self {
        serde_json::from_str(json).unwrap_or_default()
    }

    fn to_json(&self) -> String {
        serde_json::to_string(self).expect("valid json")
    }
}

#[derive(Copy, Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Configuration {
    alpha: f32,
    scale: f32,
    timestamps: bool,
    badges: bool,
    highlights: bool,
    cull_duration: u64,
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            scale: 1.0,
            timestamps: true,
            badges: true,
            highlights: true,
            cull_duration: 10,
        }
    }
}

fn load_fonts() -> FontDefinitions {
    let mut fonts = FontDefinitions::empty();

    // TODO how big is this?
    fonts.font_data.insert(
        "NotoSans-Regular".to_string(),
        FontData::from_static(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/",
            "NotoSans-Regular.ttf" // why is this a ttf
        ))),
    );

    fonts.font_data.insert(
        "NotoSansCJKjp-Regular".to_string(),
        FontData::from_static(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/",
            "NotoSansCJKjp-Regular.otf"
        ))),
    );

    fonts
        .families
        .get_mut(&FontFamily::Proportional)
        .unwrap()
        .insert(0, "NotoSans-Regular".to_string());

    fonts
        .families
        .get_mut(&FontFamily::Monospace)
        .unwrap()
        .insert(0, "NotoSans-Regular".to_string());

    fonts
        .families
        .get_mut(&FontFamily::Proportional)
        .unwrap()
        .insert(1, "NotoSansCJKjp-Regular".to_string());

    fonts
}

const CONFIGURATION_KEY: &str = concat!(env!("CARGO_PKG_NAME"), "_settings");
const HISTORY_KEY: &str = concat!(env!("CARGO_PKG_NAME"), "_history");

#[derive(gumdrop::Options)]
struct Options {
    /// Display this help message
    help: bool,
    /// Channel to join
    #[options(required)]
    channel: String,
}

fn main() -> anyhow::Result<()> {
    let opts: Options = gumdrop::parse_args_default_or_exit();

    runtime::start()?;

    eframe::run_native(
        env!("CARGO_PKG_NAME"),
        eframe::NativeOptions {
            always_on_top: true,
            decorated: false,
            transparent: true,
            mouse_passthrough: true,
            ..Default::default()
        },
        Box::new(|cc| {
            let mut config = Configuration::default();
            let mut history = History::default();

            if let Some(storage) = cc.storage {
                if let Some(data) = storage.get_string(CONFIGURATION_KEY) {
                    match serde_json::from_str(&data) {
                        Ok(stored) => config = stored,
                        Err(..) => {
                            eprintln!("ERROR: cannot load previous configuration. defaulting")
                        }
                    }
                }

                if let Some(data) = storage.get_string(HISTORY_KEY) {
                    history = History::from_json(&data)
                }
            }

            let repaint = cc.egui_ctx.clone();
            crate::runtime::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_millis(500));
                loop {
                    interval.tick().await;
                    repaint.request_repaint();
                }
            });

            cc.egui_ctx.set_pixels_per_point(config.scale);
            cc.egui_ctx.set_fonts(load_fonts());

            Box::new(OnScreenKappas::new(
                chat::start(cc.egui_ctx.clone(), opts.channel.clone()),
                ImageCache::new(ImageFetcher::spawn(cc.egui_ctx.clone())),
                BadgeMap::load_from_static_json().expect("valid static json"),
                config,
                opts.channel,
                history,
            ))
        }),
    );

    Ok(())
}
