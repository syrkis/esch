// head /////////////////////////////////////////////////////////////////////////
#import "@preview/touying:0.4.2": *
#import "@local/escher:0.0.0"

#let s = escher.register(aspect-ratio: "16-9")  // footer: self => self.info.institution)
#let s = (s.methods.info)( self: s, title: "Escher Presentation", author: "Noah Syrkis", date: datetime.today())

#let (init, slides, touying-outline, speaker-note) = utils.methods(s)
#show: init

#let (slide, empty-slide, title-slide, focus-slide) = utils.slides(s)
#show: slides


// body ////////////////////////////////////////////////////////////////////////
