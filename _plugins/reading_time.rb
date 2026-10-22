require 'cgi'

module ReadingTimeFilter
  # Estimate prose reading at 350 Chinese characters or 220 other words/minute.
  def reading_time_minutes(content)
    text = content.to_s.gsub(/<(script|style)\b[^>]*>.*?<\/\1>/mi, '')
    text = text.gsub(/\[([^\]]*)\]\([^)]*\)/, '\1')
    text = CGI.unescapeHTML(strip_html(text))
    cjk = text.scan(/[\u3400-\u9fff]/).length
    words = text.gsub(/[\u3400-\u9fff]/, ' ').scan(/[[:alnum:]]+(?:['’-][[:alnum:]]+)*/).length
    [1, (cjk / 350.0 + words / 220.0).ceil].max
  end
end

Liquid::Template.register_filter(ReadingTimeFilter)
