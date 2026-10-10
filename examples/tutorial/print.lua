-- Preserve readable Markdown while selecting vector figures and print pagination.
function Image(image)
  image.src = image.src:gsub("%.png$", ".pdf")
  if not image.src:match("04_oscillators") and not image.src:match("05_correlations") then
    image.attributes.width = "90%"
  end
  return image
end

function Header(header)
  if header.level == 1 then
    return {pandoc.RawBlock("latex", "\\clearpage"), header}
  elseif header.level == 2 then
    return {pandoc.RawBlock("latex", "\\Needspace{8\\baselineskip}"), header}
  elseif header.level == 3 then
    return {pandoc.RawBlock("latex", "\\Needspace{6\\baselineskip}"), header}
  end
end

function Math(math)
  if math.mathtype == "DisplayMath" then
    return pandoc.RawInline("latex", "\\begin{equation}" .. math.text:gsub("^%s+", ""):gsub("%s+$", "") .. "\\end{equation}")
  end
end

-- Keep a complete code excerpt together when it fits on a normal page.
function CodeBlock(block)
  local _, lines = block.text:gsub("\n", "\n")
  if lines < 65 then
    return {
      pandoc.RawBlock("latex", "\\par\\noindent\\begin{minipage}{\\linewidth}"),
      block,
      pandoc.RawBlock("latex", "\\end{minipage}\\par")
    }
  end
end
