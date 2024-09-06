// head /////////////////////////////////////////////////////////////////////////
#import "@preview/touying:0.4.2": *
#import "@local/escher:0.0.0"
#import "@preview/lovelace:0.3.0": *  // <- for pseudo code
#import "@preview/equate:0.2.0": equate  // <- for numbering equations

#let s = escher.register(aspect-ratio: "16-9")  // footer: self => self.info.institution)
#let s = (s.methods.info)( self: s, title: "Escher Presentation", author: "Noah Syrkis", date: datetime.today())
// #let s = (s.methods.enable-handout-mode)(self: s)

#let (init, slides, touying-outline, speaker-note) = utils.methods(s)
#show: init

#let (slide, empty-slide, title-slide, focus-slide) = utils.slides(s)
#show: slides

#show: equate.with(breakable: true, sub-numbering: true)
#set math.equation(numbering: "(1.1)")

// body ////////////////////////////////////////////////////////////////////////
