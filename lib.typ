#let esch(
  plot,
  xlabel: none,
  ylabel: none,
  title: none,
  caption: none,
  height: 100%,
  x-offset: 0.5em,
  y-offset: -1em,
  title-offset: -1em,
) = {
  let plot-content = box(
    // Add padding to bottom and left to account for labels
    // Add symmetric right padding when there's a ylabel to keep everything centered
    inset: (
      bottom: if xlabel != none { 1em } else { 0em },
      left: if ylabel != none { 2em } else { 0em },
      right: if ylabel != none { 2em } else { 0em },
      top: if title != none { 2em } else { 0em },
    ),
    plot
      + if title != none {
        place(top + center, dy: title-offset, text(size: 12pt, title))
      }
      + if xlabel != none {
        place(bottom + center, dy: x-offset, text(size: 12pt, xlabel))
      }
      + if ylabel != none {
        place(left + horizon, dx: y-offset, text(size: 12pt, ylabel))
      },
  )

  if caption != none {
    figure(plot-content, caption: caption)
  } else {
    plot-content
  }
}
