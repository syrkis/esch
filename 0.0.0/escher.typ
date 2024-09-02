// escher.typ
// TODO: allow or page breaks

#import "@preview/touying:0.4.2": *


#let slide(self: none, title: auto, ..args) = {
  if title != auto {
    self.bamboo-title = title
  }
  (self.methods.touying-slide)(self: self, ..args)
}

#let custom-outline(self) = {
  set text(size: 1.3em)
  set par(leading: 2em)
  locate(loc => {
    let sections = query(heading.where(level: 1), loc)
    let unique_sections = ()
    for sect in sections {
      let title = sect.body
      if title != [] and title != [References] and title not in unique_sections {
        unique_sections.push(title)
        let number = counter(heading.where(level: 1)).at(sect.location()).first()
        link(sect.location())[#number | #title]
        linebreak()
      }
    }
  })
}

#let title-slide(self: none, ..args) = {
  self = utils.empty-page(self)
  let info = self.info + args.named()
  self.page-args += (
    margin: (top: 0%, bottom: 0%, left: 5%, right: 5%),
  )
  // use the vertical bar numbering
  let body = {
    grid(
      columns: (1fr, 1fr),
      gutter: 10%,
      [
        #set align(center + horizon)
        #set par(leading: 1em)
        #block(width:100%, height: 100%)[
        #text(size: 1.4em, info.title)
        #v(1em)
        #set text(fill: self.colors.neutral-darkest)
        #if info.author != none {
          block(info.author)
        }
        #if info.date != none {
          block(if type(info.date) == datetime { info.date.display("[month repr:long] [day], [year]") } else { info.date })
        } else {
          block(" ")
        }
      ]],
      [
        #set align(left  + horizon)
        #(self.methods.touying-outline)() // TOC: VERY IMPORTANT but strange syntax
      ]
    )
  }
  (self.methods.touying-slide)(self: self, repeat: none, body)
}


#let focus-slide(self: none, body) = {
  self = utils.empty-page(self)
  self.page-args += (
    margin: 0.0em, // <-- IMPORTANT maybe make 0?
  )
  set text(size: 2em)
  (self.methods.touying-slide)(self: self, repeat: none, align(horizon + center, body))
}

#let slides(self: none, title-slide: true, slide-level: 1, ..args) = {
  if title-slide {
    (self.methods.title-slide)(self: self)
  }
  (self.methods.touying-slides)(self: self, slide-level: slide-level, ..args)
}

#let register(
  self: themes.default.register(),
  aspect-ratio: "16-9",
  footer : [],
  title: none,
  author: "Noah Syrkis",
  date: datetime.today(),
) = {
  // HEADER CONFIG
  self.escher-title = []
  let header(self) = {
    set align(top)
    show: pad.with(1.5em)
    set text(size: 30pt)
    // utils.call-or-display(self, self.escher-title)
    // states.current-section-number
    states.current-section-number(numbering: self.numbering)

    states.current-section-title

  }
  // FOOTER CONFIG
  self.escher-footer = footer
  let footer(self) = {
    set align(bottom + right)
    show: pad.with(2em)
    set text(size: 15pt)
    utils.call-or-display(self, self.escher-footer)
    states.slide-counter.display() + " of " + states.last-slide-number
  }

  // BODY CONFIG
  self.page-args += (
    paper: "presentation-" + aspect-ratio,
    header: header,
    footer: footer,
    margin: (top: 2em, bottom: 2em, left: 4em, right: 4em), // <-- IMPORTNATo position content
    // align: (horizon + center),
  )

  //  REGISTER METHODS
  self = (self.methods.numbering)(self: self, section: "1 | ", "1.1. | ") // <-- IMPORTANT but seems a little hacky

  self.methods.slide = slide
  self.methods.title-slide = title-slide
  // self.methods.new-section-slide = new-section-slide
  // self.methods.touying-new-section-slide = new-section-slide
  self.methods.focus-slide = focus-slide
  self.methods.slides = slides

  self.methods.touying-outline = (self: none, ..args) => {
    custom-outline(self)
  }
  self.methods.init = (self: none, body) => {
    set text(size: 20pt, font: "New Computer Modern")
    set list(marker: "▶")
    set par(leading: 1.5em)
    set page(paper: "presentation-" + aspect-ratio) // seems redundant (but references is vertical without)
    show heading.where(level: 1): it => {
      set text(weight: "regular")
      it
    }
    set align(horizon)
    body
    bibliography("/library.bib", title: "References", style: "ieee")
  }
  self
}
