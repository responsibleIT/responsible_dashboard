import { ComponentFixture, TestBed } from '@angular/core/testing';

import { FormInputSliderComponent } from './form-input-slider.component';

describe('FormInputSliderComponent', () => {
  let component: FormInputSliderComponent;
  let fixture: ComponentFixture<FormInputSliderComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [FormInputSliderComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(FormInputSliderComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
