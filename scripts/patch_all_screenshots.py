import os
from PIL import Image, ImageDraw, ImageFont

font_dir = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
font_bold_title = ImageFont.truetype(os.path.join(font_dir, 'segoeuib.ttf'), 15)
font_bold_hero = ImageFont.truetype(os.path.join(font_dir, 'segoeuib.ttf'), 43)
font_regular_hero = ImageFont.truetype(os.path.join(font_dir, 'segoeui.ttf'), 18)
font_icon = ImageFont.truetype(os.path.join(font_dir, 'segoeuib.ttf'), 18)
font_footer = ImageFont.truetype(os.path.join(font_dir, 'consola.ttf'), 11)

sidebar_bg = (246, 243, 238)
card_bg = (255, 254, 253)
hero_color = (28, 22, 14)
para_color = (115, 110, 102)

def patch_command_center(filepath):
    im = Image.open(filepath)
    draw = ImageDraw.Draw(im)

    # 1. Top left logo
    draw.rectangle([(68, 20), (180, 42)], fill=sidebar_bg)
    draw.text((70, 22), 'KinetixRAG', fill=(15, 23, 42), font=font_bold_title)

    # Squircle icon
    draw.rounded_rectangle([(19, 22), (53, 56)], radius=8, fill=(217, 131, 0))
    draw.text((29, 27), 'K', fill=(15, 23, 42), font=font_icon)

    # 2. Hero line 2: 'AuRAG restores the relationships.'
    draw.rectangle([(395, 285), (1200, 350)], fill=card_bg)
    draw.text((403, 290), 'KinetixRAG restores the relationships.', fill=hero_color, font=font_bold_hero)

    # 3. Hero paragraph line 2:
    draw.rectangle([(395, 395), (1200, 430)], fill=card_bg)
    draw.text((403, 399), 'disconnected systems. KinetixRAG connects them in a knowledge graph so teams can', fill=para_color, font=font_regular_hero)

    # 4. Footer: cover entire width of sidebar up to x=318
    draw.rectangle([(10, 878), (318, 912)], fill=sidebar_bg)
    draw.text((20, 886), 'KinetixRAG', fill=(115, 110, 102), font=font_footer)
    draw.text((225, 886), 'HHGoa 2026', fill=(217, 131, 0), font=font_footer)

    im.save(filepath)
    print(f'Patched {filepath}')

def patch_predictive_watch(filepath):
    im = Image.open(filepath)
    draw = ImageDraw.Draw(im)

    # 1. Top left logo
    draw.rectangle([(68, 20), (180, 42)], fill=sidebar_bg)
    draw.text((70, 22), 'KinetixRAG', fill=(15, 23, 42), font=font_bold_title)

    # Squircle icon
    draw.rounded_rectangle([(19, 22), (53, 56)], radius=8, fill=(217, 131, 0))
    draw.text((29, 27), 'K', fill=(15, 23, 42), font=font_icon)

    # 2. Footer: cover up to x=318
    draw.rectangle([(10, 878), (318, 912)], fill=sidebar_bg)
    draw.text((20, 886), 'KinetixRAG', fill=(115, 110, 102), font=font_footer)
    draw.text((225, 886), 'HHGoa 2026', fill=(217, 131, 0), font=font_footer)

    im.save(filepath)
    print(f'Patched {filepath}')

def patch_public_command_center(filepath):
    im = Image.open(filepath)
    draw = ImageDraw.Draw(im)

    # Top left logo
    draw.rectangle([(54, 16), (150, 36)], fill=sidebar_bg)
    font_sm_title = ImageFont.truetype(os.path.join(font_dir, 'segoeuib.ttf'), 13)
    draw.text((55, 18), 'KinetixRAG', fill=(15, 23, 42), font=font_sm_title)

    font_sm_icon = ImageFont.truetype(os.path.join(font_dir, 'segoeuib.ttf'), 15)
    draw.rounded_rectangle([(15, 17), (43, 45)], radius=7, fill=(217, 131, 0))
    draw.text((23, 21), 'K', fill=(15, 23, 42), font=font_sm_icon)

    # Hero line 2:
    font_sm_hero = ImageFont.truetype(os.path.join(font_dir, 'segoeuib.ttf'), 34)
    draw.rectangle([(315, 228), (960, 280)], fill=card_bg)
    draw.text((322, 231), 'KinetixRAG restores the relationships.', fill=hero_color, font=font_sm_hero)

    # Hero para line 2:
    font_sm_para = ImageFont.truetype(os.path.join(font_dir, 'segoeui.ttf'), 15)
    draw.rectangle([(315, 315), (960, 345)], fill=card_bg)
    draw.text((322, 318), 'disconnected systems. KinetixRAG connects them in a knowledge graph so teams can', fill=para_color, font=font_sm_para)

    # Footer
    draw.rectangle([(10, 860), (254, 895)], fill=sidebar_bg)
    font_sm_footer = ImageFont.truetype(os.path.join(font_dir, 'consola.ttf'), 9)
    draw.text((15, 870), 'KinetixRAG', fill=(115, 110, 102), font=font_sm_footer)
    draw.text((180, 870), 'HHGoa 2026', fill=(217, 131, 0), font=font_sm_footer)

    im.save(filepath)
    print(f'Patched {filepath}')

def patch_public_simple(filepath):
    im = Image.open(filepath)
    draw = ImageDraw.Draw(im)
    draw.rectangle([(54, 16), (150, 36)], fill=sidebar_bg)
    font_sm_title = ImageFont.truetype(os.path.join(font_dir, 'segoeuib.ttf'), 13)
    draw.text((55, 18), 'KinetixRAG', fill=(15, 23, 42), font=font_sm_title)

    font_sm_icon = ImageFont.truetype(os.path.join(font_dir, 'segoeuib.ttf'), 15)
    draw.rounded_rectangle([(15, 17), (43, 45)], radius=7, fill=(217, 131, 0))
    draw.text((23, 21), 'K', fill=(15, 23, 42), font=font_sm_icon)

    draw.rectangle([(10, 860), (254, 895)], fill=sidebar_bg)
    font_sm_footer = ImageFont.truetype(os.path.join(font_dir, 'consola.ttf'), 9)
    draw.text((15, 870), 'KinetixRAG', fill=(115, 110, 102), font=font_sm_footer)
    draw.text((180, 870), 'HHGoa 2026', fill=(217, 131, 0), font=font_sm_footer)

    im.save(filepath)
    print(f'Patched {filepath}')

def patch_dark_mm(filepath):
    im = Image.open(filepath)
    draw = ImageDraw.Draw(im)
    dark_sidebar_bg = (20, 16, 11)
    dark_footer_bg = (23, 23, 23)

    # Top left logo
    draw.rectangle([(68, 20), (180, 42)], fill=dark_sidebar_bg)
    draw.text((70, 22), 'KinetixRAG', fill=(248, 250, 252), font=font_bold_title)

    # Squircle icon
    draw.rounded_rectangle([(19, 22), (53, 56)], radius=8, fill=(217, 131, 0))
    draw.text((29, 27), 'K', fill=(15, 23, 42), font=font_icon)

    # Footer
    draw.rectangle([(10, 878), (318, 912)], fill=dark_footer_bg)
    draw.text((20, 886), 'KinetixRAG', fill=(148, 163, 184), font=font_footer)
    draw.text((225, 886), 'HHGoa 2026', fill=(217, 131, 0), font=font_footer)

    im.save(filepath)
    print(f'Patched {filepath}')

def patch_wispr_ss1(filepath):
    im = Image.open(filepath)
    draw = ImageDraw.Draw(im)

    ide_bg = (25, 25, 23)
    ide_text = (204, 204, 204)
    agent_bg = (25, 24, 26)
    prompt_bg = (42, 39, 41)

    font_ide = ImageFont.truetype(os.path.join(font_dir, 'segoeui.ttf'), 13)
    font_agent = ImageFont.truetype(os.path.join(font_dir, 'segoeuib.ttf'), 15)
    font_prompt = ImageFont.truetype(os.path.join(font_dir, 'segoeui.ttf'), 12)

    # 1. Explorer header (top left)
    draw.rectangle([(0, 48), (95, 75)], fill=ide_bg)
    draw.text((2, 52), 'kinetix hhgoa', fill=ide_text, font=font_ide)

    # 2. Agent header (right side: x: 1020..1160, y: 370..400)
    draw.rectangle([(1020, 370), (1160, 400)], fill=agent_bg)
    draw.text((1025, 375), 'kinetix hhgoa', fill=(240, 240, 240), font=font_agent)

    # 3. Prompt init line: replace AUAG with Kinetix
    draw.rectangle([(1040, 442), (1800, 465)], fill=prompt_bg)
    draw.text((1045, 444), '- Initialize a new repository for Kinetix, an autonomous industrial rag system, multi-agent', fill=(210, 210, 210), font=font_prompt)

    im.save(filepath)
    print(f'Patched {filepath}')

if __name__ == '__main__':
    patch_command_center('docs/screenshots/command-center-full.png')
    patch_command_center('docs/screenshots/command-center.png')
    patch_predictive_watch('docs/screenshots/predictive-watch-full.png')
    patch_predictive_watch('docs/screenshots/predictive-watch.png')
    patch_public_command_center('frontend/public/screenshot-command-center.png')
    patch_public_simple('frontend/public/screenshot-predictive-watch.png')
    patch_public_simple('frontend/public/screenshot-investigate.png')
    patch_dark_mm('docs/screenshots/machine-money-above-fold.png')
    patch_dark_mm('docs/screenshots/machine-money-cockpit.png')
    patch_dark_mm('docs/screenshots/machine-money-console.png')
    patch_wispr_ss1('wispr-flow/ss1.png')
    print('All screenshots patched successfully!')
