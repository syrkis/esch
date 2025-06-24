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
    inset: (
      bottom: if xlabel != none { 1em } else { 0em },
      left: if ylabel != none { 2em } else { 0em },
      top: if title != none { 2em } else { 0em },
    ),
    plot // image(image-path, height: height),
      + if title != none {
        place(top + center, dy: title-offset, title)
      }
      + if xlabel != none {
        place(bottom + center, dy: x-offset, xlabel)
      }
      + if ylabel != none {
        place(left + horizon, dx: y-offset, ylabel)
      },
  )

  if caption != none {
    figure(plot-content, caption: caption)
  } else {
    plot-content
  }
}
