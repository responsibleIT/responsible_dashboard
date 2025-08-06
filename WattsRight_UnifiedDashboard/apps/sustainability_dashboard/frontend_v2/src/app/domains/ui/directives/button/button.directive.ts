import { Directive, HostBinding, Input } from '@angular/core';

@Directive({
  selector: '[appButton]'
})
export class ButtonDirective {

  @Input() ecogreen: boolean = false;
  @Input() deeptrust: boolean = false;
  @Input() fairbeige: boolean = false;
  @Input() transparent: boolean = false;
  @Input() fullWidth: boolean = false;

  @HostBinding('class') get classes() {
    return {
      'button': true,
      'button__ecogreen': this.ecogreen,
      'button__deeptrust': this.deeptrust,
      'button__fairbeige': this.fairbeige,
      'button__transparent': this.transparent,
      'button__full-width': this.fullWidth
    };
  }
}
