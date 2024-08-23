#import "@preview/touying:0.4.2": *
#import "@local/escher:0.0.0"

#let image_path = "figs/runs/base_100_n_1024_emb_32_heads_2_depth_2_lr_0.001_epochs_800_l2_1.0_dropout_0.5/svg/"
#let title = "Mechanistic Interpretability and Implementability of Irriducible Integer Identifiers"
#let author = "Noah Syrkis"

#let s = escher.register(aspect-ratio: "16-9")  // footer: self => self.info.institution)
#let s = (s.methods.info)( self: s, title: title, author: author, date: datetime.today())
#let (init, slides, touying-outline, speaker-note) = utils.methods(s)
#show: init

#let (slide, empty-slide, title-slide) = utils.slides(s)
#show: slides

